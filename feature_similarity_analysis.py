import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from Datasets.ShipsEar_dataloader import ShipsEarDataModule
from Utils.LitModel import LitModel

# =====================================================================
# 🌟 1. 设置权重路径：脚本会自动解析该路径，并将图片存入同一文件夹
# =====================================================================
CKPT_PATH = "/root/project/HLAST_DeepShip_ParameterEfficient-main/results/ShipsEar_20260419_150217_Conservative/G1_P1_TE1_TA1_Clean/Run_0/best_model.ckpt"

def extract_features(lit_model, dataloader, device):
    lit_model.eval()
    all_features = []
    all_labels = []
    
    print("🚀 正在提取高维声纹特征...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            
            # 自动探测并调用 LitModel 中的特征提取层 (1D -> 4D Mel)
            mel_x = None
            for attr_name in ['feature_extraction', 'feature_extractor', 'mel_layer', 'mel_transform']:
                if hasattr(lit_model, attr_name):
                    mel_x = getattr(lit_model, attr_name)(x)
                    break
                    
            if mel_x is None:
                raise AttributeError("🚨 无法找到特征提取层，请检查 LitModel.py 中的变量名。")

            # 截获 Dynamic-HTAN 分类前的全局特征 [B, 256]
            features = lit_model.model(mel_x, extract_feature=True) 
            
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def plot_tsne(features, labels, class_names, save_dir):
    """
    save_dir: 图片保存的目标文件夹
    """
    save_path = os.path.join(save_dir, "tsne_feature_clustering.png")
    
    print(f"🌀 正在执行 t-SNE 降维 (目标位置: {save_dir})...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)
    
    print("🎨 正在绘制学术聚类图...")
    plt.figure(figsize=(10, 8), dpi=300) 
    
    label_names = [class_names[lbl] for lbl in labels]
    
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=label_names,
        palette="deep", 
        s=100,           
        alpha=0.85,      
        edgecolor="white",
        linewidth=0.5
    )
    
    plt.title("t-SNE Feature Clustering (Dynamic-HTAN)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title="Acoustic Classes", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 绘图成功！图片已保存至权重同级目录:\n   📂 {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 自动计算图片保存路径
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"🚨 找不到权重文件: {CKPT_PATH}")
    
    # 获取权重所在的文件夹路径
    SAVE_DIR = os.path.dirname(CKPT_PATH)
    
    # 2. 数据准备
    data_dir = 'shipsEar_AUDIOS/'
    data_module = ShipsEarDataModule(parent_folder=data_dir, batch_size={'test': 64}, num_workers=4)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    ships_classes = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])
    
    # 3. 模型加载
    dummy_params = {
        'lr': 1e-3, 'weight_decay': 1e-3, 
        'use_graph': True, 'use_prior_mask': True, 
        'use_temporal_encoder': True, 'use_temporal_attention': True
    }
    
    print(f"📦 正在加载权重: {os.path.basename(CKPT_PATH)}")
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        Params=dummy_params,
        model_name='HTAN',
        num_classes=5
    )
    lit_model.to(device)
    
    # 4. 特征提取与绘图
    features, labels = extract_features(lit_model, test_loader, device)
    plot_tsne(features, labels, class_names=ships_classes, save_dir=SAVE_DIR)