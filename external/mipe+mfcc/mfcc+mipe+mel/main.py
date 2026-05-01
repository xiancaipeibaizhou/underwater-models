import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 配置参数 ==========
TARGET_FRAMES = 100
N_MFCC = 13
N_MELS = 128               # 必须与 data_augmentation.py 中一致
MIPE_SCALES = 10
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 150
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-2
FUSION_TYPE = 'adaptive_gate_3'   # 三分支自适应门控
AUGMENT_PER_ORIGINAL = 5

# ========== 加载三种特征 ==========
print("加载增强特征...")
mfcc_feats = np.load("mfcc_augmented.npy")   # (N, T, 13, 2)
mel_feats  = np.load("mel_augmented.npy")    # (N, T, N_MELS)
mipe_seqs  = np.load("mipe_augmented.npy")   # (N, MIPE_SCALES)
labels     = np.load("labels_augmented.npy")
print(f"MFCC 形状: {mfcc_feats.shape}, Mel 形状: {mel_feats.shape}, MIPE 形状: {mipe_seqs.shape}, 标签: {labels.shape}")

# 自动计算原始音频数（假设所有原始音频产生的片段数相同）
total_augmented = len(labels)
num_original = total_augmented // (AUGMENT_PER_ORIGINAL + 1)
print(f"自动计算：原始音频数量 = {num_original}")
group_ids = []
for i in range(num_original):
    group_ids.append(i)
    for _ in range(AUGMENT_PER_ORIGINAL):
        group_ids.append(i)
group_ids = np.array(group_ids)

# 维度转换
# MFCC: (N, T, 13, 2) -> (N, 2, T, 13)
mfcc_feats = mfcc_feats.transpose(0, 3, 1, 2)
# Mel:   (N, T, N_MELS) -> (N, 1, T, N_MELS)  （单通道图像）
mel_feats = np.expand_dims(mel_feats, axis=1)   # (N, 1, T, N_MELS)

# ========== 三分支自适应门控融合 ==========
class AdaptiveGateFusion3(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, f1, f2, f3):
        concat = torch.cat([f1, f2, f3], dim=1)
        w = self.gate(concat)                 # (B,3)
        w1, w2, w3 = w[:,0:1], w[:,1:2], w[:,2:3]
        fused = w1*f1 + w2*f2 + w3*f3
        return fused, w

# ========== 三分支模型 ==========
class MFPNet3Branch(nn.Module):
    def __init__(self, mfcc_shape, mel_shape, mipe_dim, num_classes=5):
        super().__init__()
        # ----- MFCC 分支 (同原双分支) -----
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mfcc_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.mfcc_norm = nn.LayerNorm(512)
        self.mfcc_fc = nn.Linear(512, 512)

        # ----- Mel 分支 (轻量 CNN) -----
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.mel_fc = nn.Linear(128, 512)

        # ----- MIPE 分支 (同原双分支) -----
        self.mipe_embed = nn.Linear(1, 128)
        self.pos_encoder = nn.Parameter(torch.randn(1, mipe_dim, 128))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.3, batch_first=True
        )
        self.mipe_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mipe_norm = nn.LayerNorm(128)
        self.mipe_fc = nn.Linear(128, 512)

        # ----- 三分支融合 -----
        self.fusion = AdaptiveGateFusion3(feat_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, mfcc, mel, mipe):
        # MFCC
        x_m = self.mfcc_backbone(mfcc).flatten(1)
        x_m = self.mfcc_norm(x_m)
        x_m = self.mfcc_fc(x_m)
        # Mel
        x_l = self.mel_conv(mel).flatten(1)
        x_l = self.mel_fc(x_l)
        # MIPE
        x_p = mipe.unsqueeze(-1)
        x_p = self.mipe_embed(x_p)
        x_p = x_p + self.pos_encoder
        x_p = self.mipe_transformer(x_p)
        x_p = x_p.mean(dim=1)
        x_p = self.mipe_norm(x_p)
        x_p = self.mipe_fc(x_p)

        # 融合
        feat, w = self.fusion(x_m, x_l, x_p)
        return self.classifier(feat)

# ========== 统计复杂度 ==========
def print_model_complexity(model, device, mfcc_shape, mel_shape, mipe_dim):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    if THOP_AVAILABLE:
        mfcc_dummy = torch.randn(1, *mfcc_shape).to(device)
        mel_dummy  = torch.randn(1, *mel_shape).to(device)
        mipe_dummy = torch.randn(1, mipe_dim).to(device)
        flops, params = profile(model, inputs=(mfcc_dummy, mel_dummy, mipe_dummy), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs (单样本): {flops}, 参数量: {params}")
    else:
        print("未安装 thop，跳过 FLOPs 统计。")

# ========== 训练与验证函数 ==========
def train_fold(train_mfcc, train_mel, train_mipe, train_y,
               val_mfcc,   val_mel,   val_mipe,   val_y):
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_mfcc, dtype=torch.float32),
        torch.tensor(train_mel,  dtype=torch.float32),
        torch.tensor(train_mipe, dtype=torch.float32),
        torch.tensor(train_y,    dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_mfcc, dtype=torch.float32),
        torch.tensor(val_mel,  dtype=torch.float32),
        torch.tensor(val_mipe, dtype=torch.float32),
        torch.tensor(val_y,    dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = MFPNet3Branch(
        mfcc_shape=(2, TARGET_FRAMES, N_MFCC),
        mel_shape=(1, TARGET_FRAMES, N_MELS),
        mipe_dim=MIPE_SCALES,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for mf, me, mp, y in train_loader:
            mf, me, mp, y = mf.to(DEVICE), me.to(DEVICE), mp.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(mf, me, mp)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for mf, me, mp, y in val_loader:
                mf, me, mp = mf.to(DEVICE), me.to(DEVICE), mp.to(DEVICE)
                out = model(mf, me, mp)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        acc = accuracy_score(trues, preds)
        scheduler.step(avg_loss)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.3f} | Best: {best_acc:.3f}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_acc

# ========== 主程序 ==========
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    print(f"融合方式: {FUSION_TYPE} (三分支自适应门控)")

    # 打印模型复杂度
    temp_model = MFPNet3Branch(
        mfcc_shape=(2, TARGET_FRAMES, N_MFCC),
        mel_shape=(1, TARGET_FRAMES, N_MELS),
        mipe_dim=MIPE_SCALES,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print_model_complexity(temp_model, DEVICE,
                           (2, TARGET_FRAMES, N_MFCC),
                           (1, TARGET_FRAMES, N_MELS),
                           MIPE_SCALES)
    del temp_model

    print("\n开始分组5折交叉验证...")
    gkf = GroupKFold(n_splits=5)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(mfcc_feats, labels, groups=group_ids)):
        print(f"\n===== Fold {fold+1}/5 =====")
        train_mfcc = mfcc_feats[train_idx]
        train_mel  = mel_feats[train_idx]
        train_mipe = mipe_seqs[train_idx]
        train_y    = labels[train_idx]
        val_mfcc   = mfcc_feats[val_idx]
        val_mel    = mel_feats[val_idx]
        val_mipe   = mipe_seqs[val_idx]
        val_y      = labels[val_idx]

        best_acc = train_fold(train_mfcc, train_mel, train_mipe, train_y,
                              val_mfcc,   val_mel,   val_mipe,   val_y)
        fold_accs.append(best_acc)
        print(f"Fold {fold+1} 最佳准确率: {best_acc:.3f}")

    print("\n" + "="*60)
    print(f"✅ 三分支自适应门控融合 5折平均准确率: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print("="*60)