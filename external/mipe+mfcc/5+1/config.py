import torch

# ===================== 音频参数 =====================
SR = 20000                    # 论文重采样至20000 Hz
SEG_DURATION = 1.0            # 每段1秒
SEG_LEN = int(SR * SEG_DURATION)  # 20000

# ===================== 特征参数 =====================
FFT_DIM = 1000                # 20000点FFT，取前1000阶幅值
MFCC_DIM = 500                # 25个滤波器 × 20帧 = 500
CHROMA_DIM = 240              # 12个色度 × 20帧 = 240
CONTRAST_DIM = 120            # 6个对比度 × 20帧 = 120
TONNETZ_DIM = 120             # 6个音调网络特征 × 20帧 = 120
MIPE_DIM = 20                 # 将MIPE压缩到20维，使总长 = 2000
INPUT_LENGTH = 2000           # 固定输入长度

# STFT参数（用于特征提取）
N_FFT = 2048
HOP_LENGTH = 512
FRAME_LEN = 4096
HOP_LEN = 1024

# MIPE参数（重要：修改MIPE_SCALES=20，让mipe_core直接返回20维）
MIPE_SCALES = 20              # 让 multi_scale_mipe 返回20个尺度值

# ===================== 数据集 =====================
DATA_ROOT = r"E:\PyCharm\mipe+mfcc\ShipsEar"   # 请修改为您的实际路径
CLASSES = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]
NUM_CLASSES = 5

# ===================== 训练参数（与论文一致）=====================
BATCH_SIZE = 64               # 论文batch size 64
EPOCHS = 100                  # 论文epoch 100
LR = 5e-4                    # 论文学习率 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"