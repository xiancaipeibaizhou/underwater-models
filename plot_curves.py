import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 🌟 1. 设置权重路径：脚本会自动在同级目录下寻找 epoch_metrics.csv
# =====================================================================
CKPT_PATH = "/root/project/HLAST_DeepShip_ParameterEfficient-main/results/ShipsEar_20260419_150217_Conservative/G1_P1_TE1_TA1_Clean/Run_0/best_model.ckpt"

def plot_loss_curves(df, save_dir):
    """绘制并保存独立的 Loss 曲线"""
    print("📈 正在生成 Loss 曲线图...")
    plt.figure(figsize=(7, 5), dpi=300)
    
    if 'train_loss' in df.columns and 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2.5)
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#d62728', linewidth=2.5)
        
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=False, fontsize=11)
        sns.despine()
        
        save_path = os.path.join(save_dir, "loss_curves_paper.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Loss 曲线已保存至: {save_path}")
    else:
        print("⚠️ CSV 中缺少 loss 相关数据，跳过绘制。")

def plot_f1_curves(df, save_dir):
    """绘制并保存独立的 F1 指标曲线"""
    print("📈 正在生成 F1-Macro 曲线图...")
    plt.figure(figsize=(7, 5), dpi=300)
    
    if 'val_macro_f1' in df.columns:
        plt.plot(df['epoch'], df['val_macro_f1'], label='Val Macro-F1', color='#1f77b4', linewidth=2.5)
        
        # 如果有训练集 F1，也一并画出对比
        if 'train_macro_f1' in df.columns:
            plt.plot(df['epoch'], df['train_macro_f1'], label='Train Macro-F1', color='#ff7f0e', linewidth=2, alpha=0.6, linestyle='--')
        
        # 自动寻找并标注最高点
        best_idx = df['val_macro_f1'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_val = df.loc[best_idx, 'val_macro_f1']
        
        plt.scatter(best_epoch, best_val, color='red', s=80, zorder=5)
        plt.annotate(f"Best: {best_val:.4f}", 
                     xy=(best_epoch, best_val),
                     xytext=(best_epoch * 0.7, best_val - 0.08),
                     arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=.2"),
                     fontsize=11, color='red', fontweight='bold')

        plt.title('Macro-F1 Score Evolution', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Macro-F1', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=False, fontsize=11, loc='lower right')
        sns.despine()
        
        save_path = os.path.join(save_dir, "f1_curves_paper.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ F1 曲线已保存至: {save_path}")
    else:
        print("⚠️ CSV 中缺少 F1 相关数据，跳过绘制。")

if __name__ == "__main__":
    # 1. 路径校验
    if not os.path.exists(CKPT_PATH):
        print(f"🚨 找不到指定的权重文件，请检查路径: {CKPT_PATH}")
    else:
        SAVE_DIR = os.path.dirname(CKPT_PATH)
        CSV_PATH = os.path.join(SAVE_DIR, "epoch_metrics.csv")
        
        if not os.path.exists(CSV_PATH):
            print(f"🚨 在权重目录下找不到实验日志: {CSV_PATH}")
        else:
            # 2. 读取数据
            df_metrics = pd.read_csv(CSV_PATH)
            
            # 3. 执行分块绘图
            plot_loss_curves(df_metrics, SAVE_DIR)
            plot_f1_curves(df_metrics, SAVE_DIR)
            
            print("\n🎉 所有曲线图已独立生成完毕！你可以直接将它们用于论文排版。")