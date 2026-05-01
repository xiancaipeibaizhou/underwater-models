import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================
# 1. 局部时频斑块提取 (Local Spectro-Temporal Patch Extractor)
# =====================================================================
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2), pool_stride=(2, 2)):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_channels // 4), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channels // 4), nn.ReLU()
        )
        self.branch4_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.GroupNorm(1, out_channels // 4), nn.ReLU()
        )
        self.fuse_pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4_pool(x).expand(-1, -1, x.shape[2], x.shape[3]) 
        out = torch.cat([x1, x2, x3, x4], dim=1) 
        return self.fuse_pool(out)

# =====================================================================
# 2. 声学上下文感知门控 (Acoustic Context Gating)
# =====================================================================
class AcousticContextGating(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        B = x.shape[0]
        global_context = self.pool(x).view(B, -1)
        alpha = self.mlp(global_context) 
        return alpha.unsqueeze(-1) # Shape: [B, 1, 1]

# =====================================================================
# 🌟 3. PBGMR 核心：残差多头图卷积块 (RMHG-Conv Block)
# =====================================================================
class RMHG_ConvBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        
        self.proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, prior_bias=None):
        # x shape: [B*T, F_nodes, C]
        B_T, N, C = x.shape
        
        Q = self.query(x).view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scale = self.head_dim ** 0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale # [B*T, heads, N, N]
        
        if prior_bias is not None:
            # 注入物理对数先验偏置 [B*T, N, N] -> [B*T, 1, N, N]
            attn_logits = attn_logits + prior_bias.unsqueeze(1)
            
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V) 
        out = out.transpose(1, 2).contiguous().view(B_T, N, C)
        out = self.proj(out)
        
        # 强残差连接
        out = self.norm(x + F.relu(out))
        return out

# =====================================================================
# 🌟 4. 物理偏置驱动的动态谐波图 (Physics-Informed Harmonic GCN)
# =====================================================================
class HarmonicPriorGCN(nn.Module):
    def __init__(self, in_channels, num_freq_bins, num_layers=3, num_heads=4, sr=16000, fmin=20, fmax=8000, dropout=0.2): 
        super().__init__()
        self.num_freq_bins = num_freq_bins  
        self.use_prior_mask = True 
        
        self.freq_position_embedding = nn.Parameter(torch.randn(1, num_freq_bins, in_channels))
        self.prior_scale = nn.Parameter(torch.tensor(1.0)) # 自学习偏置强度
        
        # 预计算全局物理基准矩阵 (废除 TopK 裁剪)
        H_base, N_base = self._build_base_matrices(sr, fmin, fmax)
        identity = torch.eye(num_freq_bins)
        
        # 补全为双向图
        A_prior_raw = identity + N_base + H_base + H_base.transpose(0, 1)
        A_prior_norm = A_prior_raw / (A_prior_raw.sum(dim=-1, keepdim=True) + 1e-8) 
        
        # PBGMR 对数偏置
        prior_bias = torch.log(A_prior_norm + 1e-6)
        self.register_buffer("prior_bias_base", prior_bias)
        
        # 多层 RMHG 堆叠
        self.rmhg_layers = nn.ModuleList([
            RMHG_ConvBlock(in_channels, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def _build_base_matrices(self, sr, fmin, fmax, tol=0.15):
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mels = np.linspace(mel_min, mel_max, self.num_freq_bins)
        center_freqs = 700 * (10**(mels / 2595) - 1) 
        
        H_base = torch.zeros(self.num_freq_bins, self.num_freq_bins)
        N_base = torch.zeros(self.num_freq_bins, self.num_freq_bins)
        
        for i in range(self.num_freq_bins):
            for j in range(self.num_freq_bins):
                if abs(i - j) == 1:
                    N_base[i, j] = 0.5
                
                ratio = center_freqs[j] / (center_freqs[i] + 1e-8)
                for k in [2.0, 3.0, 4.0]: 
                    if abs(ratio - k) < tol:
                        weight = np.exp(- ((ratio - k)**2) / (0.05**2))
                        H_base[i, j] = max(H_base[i, j].item(), weight)
        return H_base, N_base

    def forward(self, x_cnn, dynamic_alpha=None):
        B, C, F_out, T_out = x_cnn.shape
        x_nodes = x_cnn.permute(0, 3, 2, 1).contiguous().view(B * T_out, F_out, C)
        
        freq_pos = self.freq_position_embedding.to(dtype=x_nodes.dtype)
        x_embedded = x_nodes + freq_pos
        
        dynamic_bias = None
        if self.use_prior_mask and dynamic_alpha is not None:
            alpha_expanded = dynamic_alpha.repeat_interleave(T_out, dim=0) # [B*T, 1, 1]
            p_bias = self.prior_bias_base.to(dtype=x_nodes.dtype).unsqueeze(0)
            # 自适应门控 * 模型学到的常数 * 物理图
            dynamic_bias = self.prior_scale * alpha_expanded * p_bias # [B*T, F, F]

        out = x_embedded
        for layer in self.rmhg_layers:
            out = layer(out, prior_bias=dynamic_bias)
            
        return out

# =====================================================================
# 5. 时间演化与注意力机制 (保持不变)
# =====================================================================
class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)  
        attn_weights = F.softmax(attn_weights, dim=1) 
        global_feat = torch.sum(x * attn_weights, dim=1) 
        return global_feat, attn_weights

# =====================================================================
# 6. 最终主模型：HP_STGNN (或 HTAN)
# =====================================================================
class HP_STGNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, base_channels=16, 
                 input_fdim=128, input_tdim=1024,
                 use_harmonic_graph=True, 
                 use_prior_mask=True,        
                 use_temporal_encoder=True, 
                 use_temporal_attention=True,
                 dropout=0.2):               
        super().__init__()
        
        self.use_harmonic_graph = use_harmonic_graph
        self.use_temporal_encoder = use_temporal_encoder
        self.use_temporal_attention = use_temporal_attention
        
        # 降采样池化策略 (保留 F 轴分辨率)
        self.frontend = nn.Sequential(
            MultiScaleConvBlock(in_channels, base_channels, pool_kernel=(2,2), pool_stride=(2,2)),      
            MultiScaleConvBlock(base_channels, base_channels*2, pool_kernel=(1,2), pool_stride=(1,2)), 
            MultiScaleConvBlock(base_channels*2, base_channels*4, pool_kernel=(1,2), pool_stride=(1,2)) 
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(2, in_channels, input_fdim, input_tdim) 
            self.frontend.eval()
            dummy_out = self.frontend(dummy_input)
            self.frontend.train()
            _, cnn_out_c, f_out, t_out = dummy_out.shape
            
        self.context_gating = AcousticContextGating(in_channels=cnn_out_c)
        
        # PBGMR 推荐的 RMHG 参数：3层, 4头注意力
        self.harmonic_gcn = HarmonicPriorGCN(
            in_channels=cnn_out_c, 
            num_freq_bins=f_out,
            num_layers=3,     # 🌟 RMHG 层数
            num_heads=4,      # 🌟 多头注意力数
            sr=16000, fmin=20, fmax=8000,
            dropout=dropout 
        )
        self.harmonic_gcn.use_prior_mask = use_prior_mask 
        
        gru_input_size = cnn_out_c * 2 if self.use_harmonic_graph else cnn_out_c
        self.temporal_encoder = nn.GRU(
            input_size=gru_input_size, 
            hidden_size=gru_input_size // 2, 
            num_layers=1,  
            batch_first=True, 
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttention(in_dim=gru_input_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_input_size),
            nn.Dropout(dropout), 
            nn.Linear(gru_input_size, num_classes)
        )

    def forward(self, x, extract_feature=False):  
        x_cnn = self.frontend(x)  # [B, C, F, T]
        B, C, F_out, T_out = x_cnn.shape
        
        if self.use_harmonic_graph:
            dynamic_alpha = self.context_gating(x_cnn) 
            x_g = self.harmonic_gcn(x_cnn, dynamic_alpha=dynamic_alpha) 
            x_g = x_g.view(B, T_out, F_out, C)
            
            x_mean = x_g.mean(dim=2)          
            x_max = x_g.max(dim=2).values     
            x_seq = torch.cat([x_mean, x_max], dim=-1) # [B, T, 2C]
        else:
            x_seq = x_cnn.mean(dim=2).transpose(1, 2)  
        
        if self.use_temporal_encoder:
            x_seq, _ = self.temporal_encoder(x_seq) 
            
        if self.use_temporal_attention:
            x_global, attn_weights = self.temporal_attention(x_seq) 
        else:
            x_global = x_seq.mean(dim=1)
        
        if extract_feature:
            return x_global
            
        logits = self.classifier(x_global)
        return logits