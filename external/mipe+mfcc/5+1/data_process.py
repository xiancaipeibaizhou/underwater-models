import numpy as np
import librosa
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import *
from mipe_core import multi_scale_mipe

# ===================== 特征提取函数（与之前相同） =====================
def extract_fft(y):
    n_fft = 20000
    if len(y) < n_fft:
        y_pad = np.pad(y, (0, n_fft - len(y)))
    else:
        y_pad = y[:n_fft]
    spec = np.abs(np.fft.rfft(y_pad, n=n_fft))
    # 可选：转为 dB 刻度
    spec = 20 * np.log10(spec + 1e-8)
    return spec[:FFT_DIM]

def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=25, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    if mfcc.shape[1] < 20:
        mfcc = np.pad(mfcc, ((0,0), (0, 20 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :20]
    return mfcc.flatten()[:MFCC_DIM]

def extract_chroma(y):
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if chroma.shape[1] < 20:
        chroma = np.pad(chroma, ((0,0), (0, 20 - chroma.shape[1])), mode='constant')
    else:
        chroma = chroma[:, :20]
    return chroma.flatten()[:CHROMA_DIM]

def extract_contrast(y):
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if contrast.shape[1] < 20:
        contrast = np.pad(contrast, ((0,0), (0, 20 - contrast.shape[1])), mode='constant')
    else:
        contrast = contrast[:, :20]
    return contrast.flatten()[:CONTRAST_DIM]

def extract_tonnetz(y):
    tonnetz = librosa.feature.tonnetz(y=y, sr=SR)
    if tonnetz.shape[1] < 20:
        tonnetz = np.pad(tonnetz, ((0,0), (0, 20 - tonnetz.shape[1])), mode='constant')
    else:
        tonnetz = tonnetz[:, :20]
    return tonnetz.flatten()[:TONNETZ_DIM]

def extract_mipe_feature(y):
    mipe = multi_scale_mipe(y, scales=MIPE_SCALES)   # 长度 = MIPE_SCALES (应等于 MIPE_DIM)
    if len(mipe) < MIPE_DIM:
        mipe = np.pad(mipe, (0, MIPE_DIM - len(mipe)))
    elif len(mipe) > MIPE_DIM:
        mipe = mipe[:MIPE_DIM]
    return mipe

def extract_fused_feature(y):
    fft_feat = extract_fft(y)
    mfcc_feat = extract_mfcc(y)
    chroma_feat = extract_chroma(y)
    contrast_feat = extract_contrast(y)
    tonnetz_feat = extract_tonnetz(y)
    mipe_feat = extract_mipe_feature(y)

    fused = np.concatenate([fft_feat, mfcc_feat, chroma_feat, contrast_feat, tonnetz_feat, mipe_feat])
    if len(fused) > INPUT_LENGTH:
        fused = fused[:INPUT_LENGTH]
    elif len(fused) < INPUT_LENGTH:
        fused = np.pad(fused, (0, INPUT_LENGTH - len(fused)))
    # 注意：这里不做样本内归一化，将在全局标准化时统一处理
    return fused

# ===================== 音频切分（不归一化幅值） =====================
def split_audio(y):
    """将长音频切分为1秒片段，不改变幅值"""
    segs = []
    for start in range(0, len(y), SEG_LEN):
        seg = y[start:start+SEG_LEN]
        if len(seg) < SEG_LEN:
            seg = np.pad(seg, (0, SEG_LEN - len(seg)))
        segs.append(seg)
    return segs

# ===================== 按文件级别构建数据集（防止泄露） =====================
def build_dataset():
    # 收集每个文件的所有片段 (feat, label)
    file_data = []   # 元素为 (file_path, label, [feat1, feat2, ...])
    file_labels = [] # 用于分层划分

    for class_idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_ROOT, class_name)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skip.")
            continue
        wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"Processing {class_name} ({len(wav_files)} files)...")
        for wav_file in tqdm(wav_files):
            path = os.path.join(folder, wav_file)
            y, _ = librosa.load(path, sr=SR)
            segments = split_audio(y)   # 原始音频片段
            feats = [extract_fused_feature(seg) for seg in segments]
            file_data.append((path, class_idx, feats))
            file_labels.append(class_idx)

    # 按文件划分：75% 训练，15% 验证，10% 测试
    from sklearn.model_selection import train_test_split
    file_indices = list(range(len(file_data)))
    train_idx, temp_idx = train_test_split(
        file_indices, test_size=0.25, random_state=42, stratify=file_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.4, random_state=42,
        stratify=[file_labels[i] for i in temp_idx]
    )

    # 展平片段
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for idx in train_idx:
        _, label, feats = file_data[idx]
        for feat in feats:
            X_train.append(feat)
            Y_train.append(label)
    for idx in val_idx:
        _, label, feats = file_data[idx]
        for feat in feats:
            X_val.append(feat)
            Y_val.append(label)
    for idx in test_idx:
        _, label, feats = file_data[idx]
        for feat in feats:
            X_test.append(feat)
            Y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.int64)

    # 全局标准化：只在训练集上拟合 StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存处理后的数据（包含标准化参数，但这里只保存特征数组）
    np.save("X_train.npy", X_train_scaled)
    np.save("Y_train.npy", Y_train)
    np.save("X_val.npy", X_val_scaled)
    np.save("Y_val.npy", Y_val)
    np.save("X_test.npy", X_test_scaled)
    np.save("Y_test.npy", Y_test)

    print(f"Train: {X_train_scaled.shape[0]}, Val: {X_val_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")
    print("Dataset saved with file-level split and global standardization.")

if __name__ == "__main__":
    build_dataset()