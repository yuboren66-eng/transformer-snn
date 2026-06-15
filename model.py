import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA

# ==========================================
# 1. 特征预处理与 PCA 降维 (线下/非梯度部分)
# ==========================================
class Preprocessor:
    def __init__(self, pca_ratio=0.90):
        self.pca = PCA(n_components=pca_ratio)
        self.k = None  # 降维后的特征数

    def fit_transform(self, x_np):
        """
        x_np: numpy array, 形状 [Batch, Time, Feature]
        """
        B, T, F = x_np.shape
        x_flat = x_np.reshape(-1, F)
        out_flat = self.pca.fit_transform(x_flat)
        self.k = self.pca.n_components_
        return out_flat.reshape(B, T, self.k)

    def transform(self, x_np):
        B, T, F = x_np.shape
        out_flat = self.pca.transform(x_np.reshape(-1, F))
        return out_flat.reshape(B, T, self.k)

# ==========================================
# 2. 核心网络：MLP 加权与决策级池化 (PyTorch网络)
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        
        # MLP 注意力机制：学习各个特征的权重 beta
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [Batch, Time, Feature] 此时的 Feature 是经过 PCA 降维后的 k
        """
        # --- A. 特征级融合 (MLP 注意力加权) ---
        w = x.mean(dim=1)            # [B, k] 取时间平均作为注意力的输入
        beta = self.mlp(w)           # [B, k] 生成 0~1 的权重
        x_fused = x * beta.unsqueeze(1) # [B, Time, k] 广播相乘完成加权
        
        # --- B. 决策级融合 (提取 4 维统计量) ---
        f_mean = x_fused.mean(dim=1)
        f_std  = x_fused.std(dim=1, unbiased=False)
        f_max, _ = x_fused.max(dim=1)
        f_min, _ = x_fused.min(dim=1)
        
        # 拼接成最终向量 [Batch, k * 4]
        f_final = torch.cat([f_mean, f_std, f_max, f_min], dim=-1)
        return f_final

# ==========================================
# 3. 脉冲神经网络与 Transformer 组件
# ==========================================
class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = 3.0
        s = torch.sigmoid(alpha * x)
        grad = grad_output * alpha * s * (1 - s)
        return grad

class LIFNeuron(nn.Module):
    def __init__(self, tau=2.0, threshold=1.0):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau
        self.threshold = threshold

    def forward(self, x, mem):
        mem = mem * self.decay + x
        spike = SpikeAct.apply(mem - self.threshold)
        mem = mem * (1.0 - spike.detach())   # 放电后重置
        return spike, mem

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        return x + self.pos_emb[:, :x.size(1), :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class SpikingClassifier(nn.Module):
    def __init__(self, d_model, hidden_dim, num_classes, tau=2.0, threshold=1.0, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lif = LIFNeuron(tau=tau, threshold=threshold)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, D = x.shape
        mem = None
        logits = torch.zeros(B, self.fc2.out_features, device=x.device)

        for t in range(T):
            h = self.fc1(x[:, t, :])
            h = F.gelu(self.norm(h))
            h = self.dropout(h)

            if mem is None:
                mem = torch.zeros_like(h)

            spike, mem = self.lif(h, mem)
            logits = logits + self.fc2(spike)

        return logits / T

class TransformerSNN(nn.Module):
    def __init__(
        self,
        in_dim,          
        d_model,        
        d_ff,
        n_t,
        num_classes,
        max_len,
        nhead,
        snn_hidden,
        dropout
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = LearnablePositionalEmbedding(max_len=max_len, d_model=d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, d_ff, dropout)
            for _ in range(n_t)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.snn_head = SpikingClassifier(
            d_model=d_model,
            hidden_dim=snn_hidden,
            num_classes=num_classes,
            tau=2.0,
            threshold=1.0,
            dropout=dropout
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(2)

        x = self.in_proj(x)
        x = self.pos_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = self.out_norm(x)
        return self.snn_head(x)

# ==========================================
# 4. 全局模型封装 (粘合剂：将 FusionNet 与 TransformerSNN 组合)
# ==========================================
class FullSystemModel(nn.Module):
    def __init__(self, k_dim, num_classes):
        super().__init__()
        # 1. 实例化前置融合网络
        self.fusion_net = FusionNet(in_dim=k_dim)
        
        # 2. 实例化后置分类网络
        # 巧妙的数据桥接：将 4 种统计量作为特征 (in_dim=4)，将 K 个物理特征作为序列长度 (max_len=k_dim)
        self.transformer_snn = TransformerSNN(
            in_dim=4,             # 4 维特征：[mean, std, max, min]
            d_model=32,           
            d_ff=64,
            n_t=4,                # 适当减少层数加速训练
            num_classes=num_classes,
            max_len=k_dim,        # 序列长度等于 PCA 降维后的特征数
            nhead=4,
            snn_hidden=64,
            dropout=0.1
        )

    def forward(self, x):
        # x: [Batch, Time=8, Feature=k_dim]
        
        # 1. 通过 MLP 进行特征加权和 4 维统计提取
        f_final = self.fusion_net(x) # 输出 [Batch, k_dim * 4]
        
        # 2. 维度重塑：将平铺的向量折叠为 Transformer 需要的时序格式
        # f_final 的内部排布是 [mean_1..k, std_1..k, max_1..k, min_1..k]
        B = f_final.size(0)
        K = self.fusion_net.in_dim
        
        # 变为 [Batch, 4种统计量, K个特征] 然后转置为 [Batch, K个特征(作为时序), 4种统计量(作为维度)]
        x_seq = f_final.view(B, 4, K).transpose(1, 2)
        
        # 3. 输入 Transformer-SNN
        logits = self.transformer_snn(x_seq)
        return logits

# ==========================================
# 5. 训练函数
# ==========================================
def train(model, x, y, n_train=150, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_train)

    print(f"\n--- 开始训练 (设备: {device}) ---")
    for epoch in range(n_train):
        model.train()

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == n_train - 1:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"第 {epoch+1:3d} 次循环，loss = {loss.item():.4f}, acc = {acc:.4f}")

# ==========================================
# 6. 主程序运行流水线
# ==========================================
if __name__ == "__main__":
    # ---------------------------
    # Step 0: 获取原始数据并执行 D1 判定
    # ---------------------------
    raw_matrix = [
        [-1.00187000e+01, -1.00187000e+01, -1.00187000e+01, -1.00187000e+01, -1.00187000e+01, -1.00187000e+01, -1.00187000e+01, -1.00187000e+01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 4.57863150e+03,  5.58621550e+03,  2.88826790e+03,  3.50569450e+03,  3.14584200e+03,  2.83101140e+03,  4.36853250e+03,  1.55572360e+03],
        [ 2.26449275e+04,  1.88707729e+04,  2.11352657e+04,  2.11352657e+04,  1.55572360e+03,  2.30223430e+04,  2.30223430e+04,  2.00030193e+04],
        [ 5.31315300e+02,  4.10450100e+02,  3.86978000e+02,  3.33600200e+02,  3.97614200e+02,  2.90601400e+02,  3.27107300e+02,  2.42988800e+02],
        [ 1.16000000e-01,  7.35000000e-02,  1.34000000e-01,  9.52000000e-02,  1.26400000e-01,  1.02600000e-01,  7.49000000e-02,  1.56200000e-01]
    ]
    raw_matrix = np.array(raw_matrix)
    
    s_mean = np.mean(raw_matrix[2, :])
    threshold = 1.0
    if s_mean < threshold:
        print(f"D1 判定：目标能量太低 ({s_mean:.2f})，判定为无目标，中止计算。")
        exit()
    else:
        print(f"D1 判定：检测到目标 (幅度均值 {s_mean:.2f})，进入后续识别流程。\n")

    # ---------------------------
    # Step 1: 模拟构建 Batch 数据集 (包含真实数据与加噪扩增数据)
    # ---------------------------
    # 深度学习模型需要 Batch 维度，我们模拟出 10 个样本
    batch_size = 10
    num_classes = 2
    
    dataset_list = []
    # 样本 0 使用原版真实数据，并转置为 [Time=8, Feature=6]
    dataset_list.append(raw_matrix.T) 
    # 剩余 9 个样本加入微小随机扰动进行数据增强
    for _ in range(batch_size - 1):
        noise = np.random.randn(8, 6) * 0.5
        dataset_list.append(raw_matrix.T + noise)
        
    x_batch = np.stack(dataset_list, axis=0)  # Shape: [10, 8, 6]
    
    # 模拟生成对应的 10 个分类标签 (0: 鸟类, 1: 无人机)
    y_labels = torch.randint(0, num_classes, (batch_size,))
    # 我们强制把第 0 个样本（你的真实数据）标为无人机 (1) 以供后续观察
    y_labels[0] = 1 

    # ---------------------------
    # Step 2: 线下特征预处理 (PCA)
    # ---------------------------
    prep = Preprocessor(pca_ratio=0.90)
    x_pca = prep.fit_transform(x_batch)  # Shape: [10, 8, k_dim]
    k_dim = prep.k
    print(f"PCA处理完毕: 原始特征维度 6 -> 降维后主成分维度 {k_dim}")

    # ---------------------------
    # Step 3: 实例化全局模型
    # ---------------------------
    model = FullSystemModel(k_dim=k_dim, num_classes=num_classes)
    
    # 转为 Tensor
    x_tensor = torch.tensor(x_pca, dtype=torch.float32)

    print("\n=== 数据维度流转监控 ===")
    print(f"1. 输入矩阵:     {raw_matrix.shape} -> (Feature x Time)")
    print(f"2. 批次化扩增:   {x_batch.shape} -> (Batch x Time x Feature)")
    print(f"3. PCA 降维后:   {x_tensor.shape} -> (Batch x Time x K)")
    print(f"4. 最终标签 Y:   {y_labels.shape}")

    # ---------------------------
    # Step 4: 启动联合训练
    # ---------------------------
    train(model, x_tensor, y_labels, n_train=150)

    # ---------------------------
    # Step 5: 测试与评估
    # ---------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor)
        predictions = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        print("\n==================================")
        print("★ 最终分类结果验证 ★")
        print("预测类别：", predictions.cpu().numpy())
        print("真实类别：", y_labels.cpu().numpy())
        
        print(f"\n针对你输入的真实数据 (样本 0):")
        print(f"预测为: {'无人机 (1)' if predictions[0].item() == 1 else '飞鸟/其他 (0)'}")
        print(f"模型输出置信度: {probs[0].cpu().numpy()}")
        print("==================================")