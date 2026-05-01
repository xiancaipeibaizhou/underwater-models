import numpy as np
import librosa
import os
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("paramiko").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

from tqdm import tqdm
from scipy.signal import butter, sosfilt, filtfilt
from mipe_core import multi_scale_mipe

# ========== 配置 ==========
DATA_ROOT = r"E:\PyCharm\mipe+mfcc\ShipsEar"
CLS_MAP = {"ClassA":0, "ClassB":1, "ClassC":2, "ClassD":3, "ClassE":4}
SAMPLE_RATE = 52734
SEG_SEC = 5.0
N_MFCC = 13
N_FFT = 2048
HOP_LEN = 512
TARGET_FRAMES = 100          # MFCC 和 Mel 共用时间帧数
N_MELS = 128                 # Mel 频带数，可调
MIPE_SCALES = 10
AUGMENT_PER_SEG = 5
AUGMENTATIONS = ['time_stretch', 'pitch_shift', 'add_noise', 'background_noise', 'band_mask']

# ========== 增强函数（保持不变） ==========
def time_stretch(y, rate=0.9):
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_noise(y, noise_level=0.005):
    return y + np.random.randn(len(y)) * noise_level

def background_noise(y, sr=16000, snr_db=15):
    noise = np.random.randn(len(y))
    b, a = butter(4, 0.1, 'low', fs=sr)
    noise = filtfilt(b, a, noise)
    s_p = np.mean(y**2)
    n_p = np.mean(noise**2) or 1e-10
    noise = noise * np.sqrt(s_p / (n_p * 10**(snr_db/10)))
    return y + noise

def band_mask(y, sr, fmin=200, fmax=4000):
    sos = butter(4, [fmin, fmax], 'bandstop', fs=sr, output='sos')
    return sosfilt(sos, y)

# ========== 特征提取 ==========
def extract_mfcc_delta(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
    delta = librosa.feature.delta(mfcc)
    mfcc = mfcc.T[:TARGET_FRAMES]
    delta = delta.T[:TARGET_FRAMES]
    if len(mfcc) < TARGET_FRAMES:
        pad = TARGET_FRAMES - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        delta = np.pad(delta, ((0, pad), (0, 0)))
    return np.stack([mfcc, delta], axis=-1)   # (T, 13, 2)

def extract_mel_spectrogram(y):
    """提取对数 Mel 谱图，返回 (T, N_MELS)"""
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LEN)
    log_mel = librosa.power_to_db(mel)
    log_mel = log_mel.T   # (T, N_MELS)
    if log_mel.shape[0] < TARGET_FRAMES:
        pad = TARGET_FRAMES - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)))
    else:
        log_mel = log_mel[:TARGET_FRAMES]
    return log_mel   # (T, N_MELS)

def extract_mipe_sequence(y):
    seq = multi_scale_mipe(y, scales=MIPE_SCALES)
    return np.array(seq).reshape(MIPE_SCALES,)

# ========== 5秒切分，不足补零 ==========
def split_5s(y):
    win = int(SEG_SEC * SAMPLE_RATE)
    segs = []
    for i in range(0, len(y), win):
        s = y[i:i+win]
        if len(s) < win:
            s = np.pad(s, (0, win-len(s)))
        segs.append(s)
    return segs

# ========== 遍历数据集 ==========
def get_files():
    paths, labels = [], []
    for cls, idx in CLS_MAP.items():
        folder = os.path.join(DATA_ROOT, cls)
        if not os.path.isdir(folder): continue
        for f in os.listdir(folder):
            if f.lower().endswith('.wav'):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return paths, labels

# ========== 处理单个音频 ==========
def process_file(path, label):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    segs = split_5s(y)
    out_mfcc, out_mel, out_mipe, out_lb = [], [], [], []
    for seg in segs:
        # 原始
        mf = extract_mfcc_delta(seg)
        mel = extract_mel_spectrogram(seg)
        mp = extract_mipe_sequence(seg)
        out_mfcc.append(mf)
        out_mel.append(mel)
        out_mipe.append(mp)
        out_lb.append(label)
        # 增强
        for _ in range(AUGMENT_PER_SEG):
            ty = seg.copy()
            aug = np.random.choice(AUGMENTATIONS)
            if aug == 'time_stretch': ty = time_stretch(ty, np.random.uniform(0.8,1.2))
            if aug == 'pitch_shift': ty = pitch_shift(ty, SAMPLE_RATE, np.random.randint(-3,4))
            if aug == 'add_noise': ty = add_noise(ty, np.random.uniform(0.001,0.01))
            if aug == 'background_noise': ty = background_noise(ty, SAMPLE_RATE, np.random.randint(10,25))
            if aug == 'band_mask': ty = band_mask(ty, SAMPLE_RATE, np.random.randint(100,500), np.random.randint(2000,5000))
            out_mfcc.append(extract_mfcc_delta(ty))
            out_mel.append(extract_mel_spectrogram(ty))
            out_mipe.append(extract_mipe_sequence(ty))
            out_lb.append(label)
    return out_mfcc, out_mel, out_mipe, out_lb

# ========== 主程序 ==========
if __name__ == "__main__":
    paths, labels = get_files()
    print(f"✅ 找到原始音频：{len(paths)} 个")

    all_mfcc, all_mel, all_mipe, all_lbs = [], [], [], []
    for p, l in tqdm(zip(paths, labels), total=len(paths), desc="🎵 处理音频中"):
        mf, me, mp, lb = process_file(p, l)
        all_mfcc.extend(mf)
        all_mel.extend(me)
        all_mipe.extend(mp)
        all_lbs.extend(lb)

    all_mfcc = np.array(all_mfcc)
    all_mel = np.array(all_mel)
    all_mipe = np.array(all_mipe)
    all_lbs = np.array(all_lbs)

    print(f"\n🎯 总样本数：{len(all_mfcc)}")
    print(f"📦 MFCC shape：{all_mfcc.shape}")
    print(f"📦 Mel  shape：{all_mel.shape}")
    print(f"📦 MIPE shape：{all_mipe.shape}")

    np.save("mfcc_augmented.npy", all_mfcc)
    np.save("mel_augmented.npy", all_mel)
    np.save("mipe_augmented.npy", all_mipe)
    np.save("labels_augmented.npy", all_lbs)

    print("\n✅ 全部保存完成！可直接训练")