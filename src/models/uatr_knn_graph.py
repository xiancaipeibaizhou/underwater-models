import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 轻量级 Mel Patch 提取器
# =====================================================================
class MelPatchifyBlock(nn.Module):
    def __init__(self, in_channels=1, dim=96):
        super().__init__()
        # 针对输入 [B, 1, 128, 157] 左右的 Mel 频谱
        # 经过 4 次 stride=2 下采样，空间维度变为 F/16, T/16 -> 约 8 x 10 = 80 个 Patch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            
            nn.Conv2d(48, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        out = self.stem(x) # [B, dim, F_p, T_p]
        B, C, F_p, T_p = out.shape
        # 展平为节点序列 [B, N, dim], N = F_p * T_p
        out = out.flatten(2).transpose(1, 2)
        return out, F_p, T_p

# =====================================================================
# 2. 最大相对图卷积 (Max-Relative Graph Conv)
# =====================================================================
class MRGraphConv(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
        # FFN 残差模块 (遵循论文结构)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, knn_idx):
        # x: [B, N, C], knn_idx: [B, N, K]
        B, N, C = x.shape
        K = knn_idx.size(-1)

        # 构建扁平化索引以提取邻居特征
        idx_base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_idx = (knn_idx + idx_base).reshape(-1)

        x_flat = x.reshape(B * N, C)
        neigh = x_flat[flat_idx].view(B, N, K, C) # [B, N, K, C]

        # 核心：最大相对特征 max(x_j - x_i)
        diff = neigh - x.unsqueeze(2) # [B, N, K, C]
        max_rel = diff.max(dim=2).values # [B, N, C]

        # 拼接并更新图节点
        cat_feat = torch.cat([x, max_rel], dim=-1) # [B, N, 2C]
        out = self.proj(cat_feat)
        
        # 第一次残差
        x = self.norm(x + out)  
        
        # 第二次残差 (FFN)
        out_ffn = self.ffn(x)
        x = self.ffn_norm(x + out_ffn) 
        
        return x

# =====================================================================
# 3. 最终主模型：UATR-KNN 消融框架
# =====================================================================
class UATR_KNN_Graph(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, dim=96, k=4, depth=1, variant='C', dropout=0.2):
        """
        variant: 
            'A' - Patch + Transformer (无图)
            'B' - Patch + KNN + MRGraphConv (无 Transformer)
            'C' - Patch + Transformer + KNN + MRGraphConv (完整版)
        """
        super().__init__()
        self.variant = variant
        self.dim = dim
        self.k = k
        
        self.patchify = MelPatchifyBlock(in_channels=in_channels, dim=dim)
        
        # 绝对位置编码 (支持最大 512 个 Patch，足够用了)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        if variant in ['A', 'C']:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True, norm_first=True
            )
            # L=1 or 2，保持轻量
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.norm_trans = nn.LayerNorm(dim)

        if variant in ['B', 'C']:
            self.graph_conv = MRGraphConv(dim=dim, dropout=dropout)
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def _get_knn_graph(self, x):
        dist = torch.cdist(x, x) # [B, N, N]
        # 排除自身节点
        eye = torch.eye(dist.size(1), device=dist.device).unsqueeze(0)
        dist = dist + eye * 1e6
        # 获取最近的 K 个邻居
        knn_idx = dist.topk(self.k, largest=False).indices # [B, N, K]
        return knn_idx

    def forward(self, x, extract_feature=False):
        # x: [B, 1, F, T]
        x, Fp, Tp = self.patchify(x) # x: [B, N, dim]
        B, N, C = x.shape
        
        # 加上位置编码
        x = x + self.pos_embed[:, :N, :]
        x = self.pos_drop(x)

        # === 阶段 1：全局感知 (Transformer) ===
        if self.variant in ['A', 'C']:
            x = self.transformer(x)
            x = self.norm_trans(x)

        # === 阶段 2：局部流形 (KNN + GCN) ===
        if self.variant in ['B', 'C']:
            knn_idx = self._get_knn_graph(x)
            x = self.graph_conv(x, knn_idx)

        # 全局池化
        x_pool = x.transpose(1, 2) # [B, C, N]
        x_global = self.global_pool(x_pool).squeeze(-1) # [B, C]
        
        # 兼容绘图 / SSL 提取特征
        if extract_feature:
            return x_global
            
        logits = self.classifier(x_global)
        return logits