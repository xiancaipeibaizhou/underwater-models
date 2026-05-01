import lightning as L  # 🌟 统一改成 lightning
import torch
import torch.nn.functional as F
import torch.nn as nn

class SSL_LitModel(L.LightningModule):  # 🌟 统一继承 L.LightningModule
    def __init__(self, encoder, feature_extractor, feature_dim=128, lr=1e-3):
        super().__init__()
        self.encoder = encoder  # 这就是你的 HTAN 模型
        self.feature_extractor = feature_extractor # 你的 Feature_Extraction_Layer
        self.lr = lr
        
        # 创新点1的分支 A：对比学习投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2) # 映射到低维计算对比损失
        )
        
        # 创新点1的分支 B：掩码重建解码头 (用特征粗略还原回频谱的维度，或者简单点直接在特征空间算 loss)
        # 这里为了稳妥和容易收敛，我们采用简单的“特征级重建”或“解码器重建”
        # 你可以根据频谱大小定义一个简单的反卷积或 MLP
        # 初版我们先重点把对比学习 Loss 跑通，重建 Loss 留个接口
        
    def forward(self, x):
        return self.encoder(x, extract_feature=True)

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """标准 InfoNCE 对比损失"""
        batch_size = z1.size(0)
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # 构建正样本对掩码
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / temperature)
        
        # 排除自相似
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z1.device)).float()
        denominator = mask * torch.exp(similarity_matrix / temperature)
        
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return torch.sum(loss_partial) / (2 * batch_size)

    def training_step(self, batch, batch_idx):
        # 这里的 x 是 DataLoader 传来的 1D 增强波形元组: (view1_1d, view2_1d)
        (w1, w2), _ = batch 
        
        # 1. 经过物理特征提取，转化为 2D 梅尔谱并施加物理掩码 (Time/Freq Masking)
        # return_clean=True 让我们既拿到了被掩蔽的谱，又拿到了干净谱（如果后续要加重建Loss）
        spec1_masked, spec1_clean = self.feature_extractor(w1, return_clean=True)
        spec2_masked, spec2_clean = self.feature_extractor(w2, return_clean=True)
        
        # 2. 经过 HTAN 提取深度时频特征
        feat1 = self.encoder(spec1_masked, extract_feature=True)
        feat2 = self.encoder(spec2_masked, extract_feature=True)
        
        # 3. 经过对比头计算
        z1 = self.projector(feat1)
        z2 = self.projector(feat2)
        
        # 4. 计算对比损失
        contrastive_loss = self.nt_xent_loss(z1, z2)
        
        # (可选) 5. 如果要加上频谱重建 Loss，可以在这里算 spec1_clean 和某个 decoder(feat1) 的 MSE
        # recon_loss = F.mse_loss(self.decoder(feat1), spec1_clean)
        
        total_loss = contrastive_loss # + 0.1 * recon_loss
        
        self.log('ssl_loss', total_loss, on_step=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer