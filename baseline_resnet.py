import os
import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import F1Score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 复用你现有的数据模块
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

# ==========================================
# 绘图工具函数 (高标准论文级混淆矩阵)
# ==========================================
def plot_and_save_confusion_matrix(cm, target_names, save_path):
    clean_target_names = [str(name).replace('\x96', '-').replace('\u2013', '-') for name in target_names]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) 
    
    num_classes = len(clean_target_names)
    fig_width, fig_height = max(8, num_classes * 1.2), max(6, num_classes * 1.0)
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=1.1) 
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else "0"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap='Blues', cbar=True,
                xticklabels=clean_target_names, yticklabels=clean_target_names, vmin=0.0, vmax=1.0)
    
    plt.title('Baseline ResNet18 Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 核心：Baseline Lightning Module
# ==========================================
class Baseline_ResNet_LitModel(L.LightningModule):
    def __init__(self, num_classes=5, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # 1. 极简声学特征提取 (Log-Mel)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=64
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

        # 2. 经典 ResNet18 主干
        self.model = models.resnet18(weights=None)
        
        # 修改第一层：接收 1 通道
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改最后一层：输出 5 类
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        
        # 🌟 注册验证集 F1 计算器
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        # x shape: [B, T_samples]
        mel = self.mel_transform(x)        # [B, 64, T_frames]
        log_mel = self.db_transform(mel)   # [B, 64, T_frames]
        
        # 极简标准化: (x - mean) / std (per-sample)
        mean = log_mel.mean(dim=[-2, -1], keepdim=True)
        std = log_mel.std(dim=[-2, -1], keepdim=True)
        log_mel_norm = (log_mel - mean) / (std + 1e-6)

        # 增加 Channel 维度喂给 CNN: [B, 1, 64, T_frames]
        log_mel_norm = log_mel_norm.unsqueeze(1)
        
        logits = self.model(log_mel_norm)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # 🌟 更新并记录 F1 和 Loss
        self.val_macro_f1(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_macro_f1', self.val_macro_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_targets = []
            
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        return self.criterion(logits, y)

    def on_test_epoch_end(self):
        if hasattr(self, 'test_preds') and len(self.test_preds) > 0:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            
            acc = accuracy_score(targets, preds)
            f1_mac = f1_score(targets, preds, average='macro', zero_division=0)
            
            self.custom_metrics = {
                'Baseline_ACC': acc,
                'Baseline_F1_Macro': f1_mac
            }
            
            save_dir = getattr(self, "test_save_dir", ".")
            class_names = getattr(self, "class_names", [f"Class {i}" for i in range(self.num_classes)])
            
            cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
            plot_and_save_confusion_matrix(cm, class_names, os.path.join(save_dir, "baseline_confusion_matrix.png"))
            
            report = classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0)
            with open(os.path.join(save_dir, "baseline_classification_report.txt"), "w") as f:
                f.write(report)
            
            print("\n" + "="*60)
            print("🚀 [BASELINE TEST SET] DETAILED CLASSIFICATION REPORT")
            print("="*60)
            print(report)
            print("="*60)
            
            self.test_preds.clear()
            self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 🌟 调度器盯着 val_macro_f1，模式设为 max
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_macro_f1"}
        }

# ==========================================
# 主运行脚本
# ==========================================
def main():
    print("🚀 启动 ShipsEar 最纯粹 Baseline: ResNet18 + Log-Mel ...")
    L.seed_everything(42)

    # 1. 初始化数据 (严格复用你现有的数据路径，关闭 SSL)
    datamodule = ShipsEarDataModule(
        parent_folder='./shipsEar_AUDIOS', 
        batch_size={'train': 32, 'val': 32, 'test': 32}, # 如果显存不够可降到 16
        num_workers=8,
        is_ssl=False  
    )
    datamodule.setup()

    # 2. 实例化模型
    model = Baseline_ResNet_LitModel(num_classes=5, lr=1e-3)

    # 3. 设置回调
    # 🌟 保存最佳 F1 权重的回调
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/baseline/',
        filename='best_resnet-{epoch:02d}-{val_macro_f1:.4f}',
        monitor='val_macro_f1',
        mode='max',
        save_top_k=1,
    )
    
    # 🌟 早停回调 (盯着 F1)
    early_stop_callback = EarlyStopping(
        monitor='val_macro_f1', min_delta=0.00, patience=10, verbose=True, mode='max'
    )

    csv_logger = CSVLogger("results/baseline_results", name="training_logs")

    # 4. 开始训练
    trainer = L.Trainer(
        max_epochs=80, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger
    )

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    
    # 5. 测试与保存
    print("\n🧪 Baseline 训练完成，开始 Test 集评测...")
    res_dir = './results/baseline_results'
    os.makedirs(res_dir, exist_ok=True)
    
    model.test_save_dir = res_dir
    model.class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'] 
    
    trainer.test(model, dataloaders=datamodule.test_dataloader(), ckpt_path='best')
    
    if hasattr(model, 'custom_metrics'):
        df = pd.DataFrame([model.custom_metrics])
        csv_path = os.path.join(res_dir, 'baseline_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"📊 Baseline 指标已保存至 CSV: {csv_path}")

if __name__ == '__main__':
    main()