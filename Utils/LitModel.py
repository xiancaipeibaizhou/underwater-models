"""PyTorch Lightning 模型封装。

LitModel 统一管理不同 model_name 的构建、Log-Mel 特征提取、
训练/验证/测试指标和 checkpoint 加载时的重建逻辑。

除 MF_* 多特征模型外，大多数模型都直接接收 waveform，然后先经过
Feature_Extraction_Layer 得到 Log-Mel，再送入具体模型主体。
"""

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import F1Score
from src.models.custom_model import HP_STGNN
from src.models.uatr_knn_reg import AcousticAuxTargetExtractor, UATR_KNN_REG
from src.models.uatr_knn_graph import UATR_KNN_Graph  # 🌟 新增：导入轻量化模型
from src.models.fa_uatr_knn import FA_UATR_KNN
from src.models.shufflefac import ShuffleFAC
from src.models.stereo_semantic_net import KnowledgeUpdateStereoSemanticNet
from src.models.multifeature_fusion import (
    MultiFeatureConcatMLP,
    MultiFeatureBranchFusion,
    MultiFeatureCrossAttention,
    MultiViewKNNFusion,
)
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer

# ==========================================
# 引入 sklearn 与 seaborn，严格仿写 MILAN 可视化
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

MULTIFEATURE_MODELS = ['MF_CONCAT', 'MF_BRANCH', 'MF_CROSSATTN', 'MF_KNN']

def plot_and_save_confusion_matrix(cm, target_names, save_path):
    """MILAN 同款高级混淆矩阵画图函数 (论文级)"""
    clean_target_names = [str(name).replace('\x96', '-').replace('\u2013', '-') for name in target_names]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) 
    
    num_classes = len(clean_target_names)
    fig_width = max(8, num_classes * 1.2)
    fig_height = max(6, num_classes * 1.0)
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=1.1) 
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 过滤掉 0，保持画面极度整洁
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else "0"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap='Blues', cbar=True,
                xticklabels=clean_target_names, yticklabels=clean_target_names, vmin=0.0, vmax=1.0)
    
    plt.title('Normalized Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


class LitModel(L.LightningModule):
    """统一的 LightningModule。

    model_name 对应关系：
    - HTAN: 物理启发频率图网络路线。
    - UATR_KNN: Log-Mel -> Patch/Transformer/KNN-GNN。
    - ShuffleFAC: Log-Mel -> FA/FASC 轻量 CNN。
    - FA_UATR_KNN: 后续融合路线，Log-Mel -> FASC -> UATR_KNN 风格关系建模。
    - StereoSemanticNet: 知识嵌入语义网络。
    - MF_*: 人工特征多视图路线，直接由模型内部处理 waveform/dict。
    - UATR_KNN_REG: Log-Mel 主干 + 人工特征辅助回归目标。
    """
    def __init__(self, Params, model_name, num_classes, numBins=None, RR=None):
        super().__init__()
        self.save_hyperparameters()
        self.Params = Params
        self.num_classes = num_classes
        self.model_name = model_name

        if self.model_name in MULTIFEATURE_MODELS:
            self.feature_extractor = None
        else:
            self.feature_extractor = Feature_Extraction_Layer(
                input_feature=Params.get('audio_feature', 'LogMelFBank'),
                sample_rate=Params.get('sample_rate', 16000),
                window_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                number_mels=Params.get('number_mels', 128),
                segment_length=Params.get('segment_length', 5)
            )

        def safe_get(key, default=True):
            if hasattr(Params, key): return getattr(Params, key)
            if isinstance(Params, dict): return Params.get(key, default)
            return default

        # 下面的分支只负责选择模型主体。除 MULTIFEATURE_MODELS 和
        # UATR_KNN_REG 的特殊辅助头外，forward 都会统一执行：
        # waveform -> Feature_Extraction_Layer -> self.model(logmel)。
        if self.model_name == 'HTAN':
            expected_t_dim = int((Params.get('segment_length', 5) * Params.get('sample_rate', 16000)) / Params.get('hop_length', 512)) + 1
            
            self.model = HP_STGNN( 
                num_classes=self.num_classes,
                in_channels=1, 
                base_channels=Params.get('base_channels', 16), 
                input_fdim=Params.get('number_mels', 128),
                input_tdim=expected_t_dim,
                use_harmonic_graph=safe_get('use_graph', True),
                use_prior_mask=safe_get('use_prior_mask', True), 
                use_temporal_encoder=safe_get('use_temporal_encoder', True),
                use_temporal_attention=safe_get('use_temporal_attention', True),
                dropout=Params.get('dropout', 0.2)               
            )
            
        elif self.model_name == 'UATR_KNN':  
            v = Params.get('uatr_variant', 'C').upper()
                
            print(f"🔥 加载 UATR_KNN_Graph, 变体类型: 实验 {v}")
            self.model = UATR_KNN_Graph(
                num_classes=self.num_classes,
                in_channels=1,
                dim=96,
                k=8,          
                depth=1,      
                variant=v,
                dropout=Params.get('dropout', 0.2)
            )

        elif self.model_name == 'UATR_KNN_REG':
            n_mfcc = Params.get('n_mfcc', 20)
            stft_bins = Params.get('stft_bins', 64)
            computed_aux_dim = (n_mfcc * 4) + (stft_bins * 2)
            aux_target_dim = Params.get('aux_target_dim', computed_aux_dim)
            if aux_target_dim != computed_aux_dim:
                raise ValueError(
                    f"aux_target_dim must match n_mfcc/stft_bins ({computed_aux_dim}), "
                    f"got {aux_target_dim}."
                )

            print("Loading UATR_KNN_REG: Log-Mel UATR_KNN-C + handcrafted auxiliary regression")
            self.model = UATR_KNN_REG(
                num_classes=self.num_classes,
                in_channels=1,
                dim=96,
                k=8,
                depth=1,
                dropout=Params.get('dropout', 0.2),
                aux_target_dim=aux_target_dim,
                aux_hidden_dim=Params.get('aux_hidden_dim'),
            )
            self.aux_target_extractor = AcousticAuxTargetExtractor(
                sample_rate=Params.get('sample_rate', 16000),
                n_mfcc=n_mfcc,
                n_fft=Params.get('n_fft', Params.get('window_length', 2048)),
                win_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                n_mels=Params.get('number_mels', 128),
                stft_bins=stft_bins,
            )
            self.aux_loss_weight = Params.get('aux_loss_weight', 0.05)
            self.aux_criterion = nn.SmoothL1Loss()

        elif self.model_name == 'ShuffleFAC':
            print("Loading ShuffleFAC gamma=16 on Log-Mel input")
            self.model = ShuffleFAC(
                num_classes=self.num_classes,
                in_channels=1,
                n_mels=Params.get('number_mels', 128),
                activation=Params.get('shufflefac_activation', 'glu'),
                dropout=Params.get('dropout', 0.2),
                filters=[16, 32, 64, 128, 128, 128, 128],
            )

        elif self.model_name == 'FA_UATR_KNN':
            print("Loading FA_UATR_KNN: FASC stem + Transformer + KNN-GNN gated fusion")
            self.model = FA_UATR_KNN(
                num_classes=self.num_classes,
                in_channels=1,
                dim=Params.get('fusion_dim', 128),
                k=Params.get('knn_k', 8),
                depth=Params.get('uatr_depth', 1),
                dropout=Params.get('dropout', 0.2),
                n_mels=Params.get('number_mels', 128),
            )

        elif self.model_name == 'StereoSemanticNet':
            print("🔥 加载基于知识嵌入的立体语义网络 (StereoSemanticNet)")
            self.model = KnowledgeUpdateStereoSemanticNet(
                num_classes=self.num_classes,
                feature_dim=128,
                hidden_dim=64
            )

        elif self.model_name == 'MF_CONCAT':
            self.model = MultiFeatureConcatMLP(
                num_classes=self.num_classes,
                sample_rate=Params.get('sample_rate', 16000),
                n_mfcc=Params.get('n_mfcc', 20),
                n_fft=Params.get('n_fft', Params.get('window_length', 2048)),
                win_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                n_mels=Params.get('number_mels', 128),
                mipe_m=Params.get('mipe_m', 3),
                mipe_tau=Params.get('mipe_tau', 1),
                mipe_c=Params.get('mipe_c', 10),
                mipe_scale=Params.get('mipe_scale', 10),
                disable_mipe=Params.get('disable_mipe', False),
                require_cached_mipe=Params.get('use_cached_mipe', False),
                dropout=Params.get('dropout', 0.2),
            )

        elif self.model_name == 'MF_BRANCH':
            self.model = MultiFeatureBranchFusion(
                num_classes=self.num_classes,
                sample_rate=Params.get('sample_rate', 16000),
                n_mfcc=Params.get('n_mfcc', 20),
                n_fft=Params.get('n_fft', Params.get('window_length', 2048)),
                win_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                n_mels=Params.get('number_mels', 128),
                mipe_m=Params.get('mipe_m', 3),
                mipe_tau=Params.get('mipe_tau', 1),
                mipe_c=Params.get('mipe_c', 10),
                mipe_scale=Params.get('mipe_scale', 10),
                disable_mipe=Params.get('disable_mipe', False),
                require_cached_mipe=Params.get('use_cached_mipe', False),
                d_model=Params.get('fusion_dim', 128),
                dropout=Params.get('fusion_dropout', Params.get('dropout', 0.2)),
            )

        elif self.model_name == 'MF_CROSSATTN':
            self.model = MultiFeatureCrossAttention(
                num_classes=self.num_classes,
                sample_rate=Params.get('sample_rate', 16000),
                n_mfcc=Params.get('n_mfcc', 20),
                n_fft=Params.get('n_fft', Params.get('window_length', 2048)),
                win_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                n_mels=Params.get('number_mels', 128),
                mipe_m=Params.get('mipe_m', 3),
                mipe_tau=Params.get('mipe_tau', 1),
                mipe_c=Params.get('mipe_c', 10),
                mipe_scale=Params.get('mipe_scale', 10),
                disable_mipe=Params.get('disable_mipe', False),
                require_cached_mipe=Params.get('use_cached_mipe', False),
                d_model=Params.get('fusion_dim', 128),
                dropout=Params.get('fusion_dropout', Params.get('dropout', 0.2)),
            )

        elif self.model_name == 'MF_KNN':
            self.model = MultiViewKNNFusion(
                num_classes=self.num_classes,
                sample_rate=Params.get('sample_rate', 16000),
                n_mfcc=Params.get('n_mfcc', 20),
                n_fft=Params.get('n_fft', Params.get('window_length', 2048)),
                win_length=Params.get('window_length', 2048),
                hop_length=Params.get('hop_length', 512),
                n_mels=Params.get('number_mels', 128),
                mipe_m=Params.get('mipe_m', 3),
                mipe_tau=Params.get('mipe_tau', 1),
                mipe_c=Params.get('mipe_c', 10),
                mipe_scale=Params.get('mipe_scale', 10),
                disable_mipe=Params.get('disable_mipe', False),
                require_cached_mipe=Params.get('use_cached_mipe', False),
                d_model=Params.get('fusion_dim', 128),
                k=Params.get('knn_k', 4),
                dropout=Params.get('fusion_dropout', Params.get('dropout', 0.2)),
            )
               
        else:
            raise ValueError(f"Unsupported model: {model_name}. Please use HTAN, UATR_KNN, ShuffleFAC, FA_UATR_KNN, StereoSemanticNet, or MF_* models.")

        self.criterion = nn.CrossEntropyLoss()
        self.aux_target_extractor = getattr(self, "aux_target_extractor", None)
        self.aux_loss_weight = float(getattr(self, "aux_loss_weight", Params.get('aux_loss_weight', 0.05)))
        self.aux_criterion = getattr(self, "aux_criterion", nn.SmoothL1Loss())
        
        # 仅保留用于 EarlyStopping 监控的验证集 F1
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        
        # 拦截器：收集预测和标签以交给 sklearn 处理
        self.test_preds = []
        self.test_targets = []
        self.test_probs = []
        self.test_paths = []
        
        # 🌟 动态判定数据集的 class names
        if self.num_classes == 5:
            self.class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']  # ShipsEar
        elif self.num_classes == 4:
            self.class_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']              # DeepShip
        else:
            self.class_names = [f'Class {i}' for i in range(self.num_classes)]

    def forward(self, x):
        """前向传播。

        常规模型输入 waveform，先提取 Log-Mel；MF_* 模型由自身处理输入；
        UATR_KNN_REG 需要同时返回辅助回归分支，因此单独分流。
        """
        if self.model_name in MULTIFEATURE_MODELS:
            return self.model(x)
        if self.model_name == 'UATR_KNN_REG':
            return self._forward_uatr_reg(x, return_aux=False)
        if isinstance(x, dict):
            x = x.get("waveform", x.get("x"))
        x = self.feature_extractor(x)
        return self.model(x)

    def _unpack_batch(self, batch):
        """兼容普通 Dataset 与带路径/多特征 dict 的 batch 格式。"""
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch[0], batch[1], batch[2]

        x, y = batch
        paths = None
        if isinstance(x, dict) and "path" in x:
            paths = x["path"]
            x = dict(x)
            x.pop("path", None)
        return x, y, paths

    def _normalize_paths(self, paths, batch_size):
        if paths is None:
            return [""] * batch_size
        if isinstance(paths, str):
            return [paths]
        if isinstance(paths, (list, tuple)):
            return [str(p) for p in paths]
        return [str(p) for p in list(paths)]

    def _extract_waveform(self, x):
        if isinstance(x, dict):
            waveform = x.get("waveform")
            if waveform is None:
                waveform = x.get("x")
            if waveform is None:
                raise ValueError("Input dict must contain 'waveform'.")
            return waveform
        return x

    def _forward_uatr_reg(self, x, return_aux=False):
        waveform = self._extract_waveform(x)
        log_mel = self.feature_extractor(waveform)
        return self.model(log_mel, return_aux=return_aux)

    def _build_aux_target(self, x):
        if self.aux_target_extractor is None:
            raise RuntimeError("aux_target_extractor is not initialized.")
        with torch.no_grad():
            waveform = self._extract_waveform(x)
            return self.aux_target_extractor(waveform)

    def _log_gate_mean(self):
        if hasattr(self.model, "last_gate_mean") and self.model.last_gate_mean is not None:
            self.log("gate_mean", self.model.last_gate_mean, on_step=False, on_epoch=True, prog_bar=False, logger=False)

    def training_step(self, batch, batch_idx):
        """训练 step。

        常规模型只计算交叉熵；UATR_KNN_REG 额外加入辅助回归损失。
        如果模型暴露 last_gate_mean，会记录 gate_mean 便于观察图分支权重。
        """
        x, y, _ = self._unpack_batch(batch)
        if self.model_name == 'UATR_KNN_REG':
            logits, aux_pred = self._forward_uatr_reg(x, return_aux=True)
            ce_loss = self.criterion(logits, y)
            aux_target = self._build_aux_target(x).to(device=aux_pred.device, dtype=aux_pred.dtype)
            aux_loss = self.aux_criterion(aux_pred, aux_target)
            total_loss = ce_loss + self.aux_loss_weight * aux_loss

            self.log('ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.log('aux_loss', aux_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.log('total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return total_loss

        logits = self(x)
        loss = self.criterion(logits, y)
        self._log_gate_mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False) # 关掉 logger
        return loss

    def validation_step(self, batch, batch_idx):
        """验证 step。

        早停和 checkpoint 监控 `val_macro_f1`，因此这里维护 torchmetrics
        的 macro-F1，同时记录验证 loss。
        """
        x, y, _ = self._unpack_batch(batch)
        if self.model_name == 'UATR_KNN_REG':
            logits, aux_pred = self._forward_uatr_reg(x, return_aux=True)
            ce_loss = self.criterion(logits, y)
            aux_target = self._build_aux_target(x).to(device=aux_pred.device, dtype=aux_pred.dtype)
            aux_loss = self.aux_criterion(aux_pred, aux_target)
            loss = ce_loss + self.aux_loss_weight * aux_loss
            self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.log('val_aux_loss', aux_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.log('val_total_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
            self._log_gate_mean()
        preds = torch.argmax(logits, dim=1)
        
        self.val_macro_f1(preds, y) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        """测试 step。

        不在 step 内直接计算最终指标，而是缓存预测、标签、概率和路径，
        交给 on_test_epoch_end 使用 sklearn 一次性计算。
        """
        x, y, paths = self._unpack_batch(batch)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        
        # 拦截并存起来，不在 Lightning 内部算乱七八糟的指标
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        self.test_probs.append(probs.detach().cpu())
        self.test_paths.extend(self._normalize_paths(paths, y.size(0)))
        
        return self.criterion(logits, y)

    def on_test_epoch_end(self):
        """使用 sklearn 计算最终测试指标并保存报告。

        输出包括 ACC、weighted precision/recall、Macro-F1、Weighted-F1、
        confusion_matrix.png、classification_report.txt 和 test_predictions.csv。
        """
        if len(self.test_preds) > 0:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            probs = torch.cat(self.test_probs).numpy()
            
            # 1. 算尽天下指标
            acc = accuracy_score(targets, preds)
            apr = precision_score(targets, preds, average='weighted', zero_division=0)
            re = recall_score(targets, preds, average='weighted', zero_division=0)
            f1_mac = f1_score(targets, preds, average='macro', zero_division=0)
            f1_wei = f1_score(targets, preds, average='weighted', zero_division=0)
            
            # 将指标抛出给外部的 demo_light.py 拿去写 CSV
            self.custom_metrics = {
                'ACC': acc,
                'APR_Weighted': apr,
                'RE_Weighted': re,
                'F1_Macro': f1_mac,
                'F1_Weighted': f1_wei
            }
            
            # 2. 定位我们在主函数指定的专属极简文件夹
            save_dir = getattr(self, "test_save_dir", ".")
            
            # 3. 绘制 SOTA 级混淆矩阵
            cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
            plot_and_save_confusion_matrix(cm, self.class_names, os.path.join(save_dir, "confusion_matrix.png"))
            
            # 4. 打印极其华丽的分类报告
            report = classification_report(
                targets,
                preds,
                labels=list(range(self.num_classes)),
                target_names=self.class_names,
                digits=4,
                zero_division=0
            )
            with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
                f.write(report)

            pred_path = os.path.join(save_dir, "test_predictions.csv")
            fieldnames = ["sample_id", "file_path", "true_label", "pred_label", "confidence"]
            fieldnames.extend([f"prob_class_{i}" for i in range(self.num_classes)])
            with open(pred_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for sample_id, (target, pred, prob_row) in enumerate(zip(targets, preds, probs)):
                    row = {
                        "sample_id": sample_id,
                        "file_path": self.test_paths[sample_id] if sample_id < len(self.test_paths) else "",
                        "true_label": int(target),
                        "pred_label": int(pred),
                        "confidence": float(np.max(prob_row)),
                    }
                    for class_idx in range(self.num_classes):
                        row[f"prob_class_{class_idx}"] = float(prob_row[class_idx])
                    writer.writerow(row)
            
            print("\n" + "="*60)
            print("🚀 [TEST SET] DETAILED CLASSIFICATION REPORT")
            print("="*60)
            print(report)
            print("="*60)
            print(f"✅ Metrics & Confusion Matrix Image accurately saved to:\n   {save_dir}\n")
            
            # 清空内存防 OOM
            self.test_preds.clear()
            self.test_targets.clear()
            self.test_probs.clear()
            self.test_paths.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.Params['lr'], 
            weight_decay=self.Params.get('weight_decay', 1e-5)  # 👈 从超参数字典获取
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_macro_f1", 
                "frequency": 1
            },
        }
