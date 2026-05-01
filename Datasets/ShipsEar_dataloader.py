"""DeepShip / ShipsEar 分段音频 DataModule。

本文件同时服务 DeepShip 预切片目录和 ShipsEar 预切片目录。核心区别是：
- frame-level split: 以切片为单位随机划分，可能产生同录音 overlap。
- recording-level split: 以录音文件夹为单位划分，再展开为切片，是正式实验协议。

setup 后会输出 recording overlap audit；recording-level 模式下 overlap 非 0 会报错。
"""

import os
import json
import collections
from datetime import datetime
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import hashlib  
import random  
from Utils.MultiFeature_Extraction_Layer import MultiFeatureExtractor

def add_awgn(signal, snr_db, seed_string=None):
    """根据目标信噪比 (SNR) 注入绝对固定的高斯白噪声"""
    if snr_db is None:
        return signal
        
    sig_power = np.mean(signal ** 2)
    if sig_power == 0:
        return signal
        
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    if seed_string is not None:
        seed = int(hashlib.md5(seed_string.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    noise = rng.normal(0, np.sqrt(noise_power), len(signal)).astype(np.float32)
    return signal + noise


class ShipsEarDataset(Dataset):
    """读取单个音频切片并返回 waveform/label。

    默认返回 `(waveform, target)`；测试集可通过 return_path=True 返回
    `({"waveform": waveform, "path": file_path}, target)`，方便保存预测明细。
    如果 use_cached_mipe=True，会额外返回缓存/即时计算的 MIPE 特征给保留的
    多特征路线使用；Log-Mel 主线模型不依赖该字段。
    """
    def __init__(
        self,
        segment_list,
        target_sr=16000,
        normalize_waveform=False,
        snr_db=None,
        is_ssl=False,
        use_cached_mipe=False,
        mipe_cache_dir=None,
        mipe_cache_root=None,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        return_path=False,
    ):
        self.segment_list = segment_list
        self.target_sr = target_sr
        self.normalize_waveform = normalize_waveform
        self.snr_db = snr_db
        self.is_ssl = is_ssl
        self.use_cached_mipe = use_cached_mipe and not disable_mipe and not is_ssl
        self.mipe_cache_dir = mipe_cache_dir
        self.mipe_cache_root = mipe_cache_root
        self.mipe_m = mipe_m
        self.mipe_tau = mipe_tau
        self.mipe_c = mipe_c
        self.mipe_scale = mipe_scale
        self.disable_mipe = disable_mipe
        self.return_path = return_path
        self._mipe_extractor = None
        self._mipe_config = {
            "target_sr": target_sr,
            "normalize_waveform": normalize_waveform,
            "snr_db": snr_db,
            "mipe_m": mipe_m,
            "mipe_tau": mipe_tau,
            "mipe_c": mipe_c,
            "mipe_scale": mipe_scale,
        }

    def __len__(self):
        return len(self.segment_list)

    def _read_waveform(self, file_path):
        try:
            sample_rate, signal = wavfile.read(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file: {file_path}. Details: {e}")

        if sample_rate != self.target_sr:
            raise ValueError(f"Expected {self.target_sr}Hz, got {sample_rate}Hz for {file_path}")

        if len(signal) == 0:
            raise ValueError(f"Empty audio file: {file_path}")

        signal = signal.astype(np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        return signal

    def _prepare_supervised_waveform(self, file_path):
        signal = self._read_waveform(file_path)
        signal = add_awgn(signal, self.snr_db, seed_string=file_path)

        if self.normalize_waveform:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
        return signal.astype(np.float32)

    def _get_mipe_extractor(self):
        if self._mipe_extractor is None:
            self._mipe_extractor = MultiFeatureExtractor(
                sample_rate=self.target_sr,
                mipe_m=self.mipe_m,
                mipe_tau=self.mipe_tau,
                mipe_c=self.mipe_c,
                mipe_scale=self.mipe_scale,
                disable_mipe=False,
            )
        return self._mipe_extractor

    def _snr_cache_suffix(self):
        if self.snr_db is None:
            return ".mipe.pt"
        snr = str(self.snr_db).replace("-", "m").replace(".", "p")
        return f".snr{snr}.mipe.pt"

    def _mipe_cache_path(self, file_path):
        suffix = self._snr_cache_suffix()
        if self.mipe_cache_dir:
            root = self.mipe_cache_root or os.path.commonpath([p for p, _ in self.segment_list])
            rel_path = os.path.relpath(file_path, root)
            rel_base = os.path.splitext(rel_path)[0] + suffix
            return os.path.join(self.mipe_cache_dir, rel_base)
        return os.path.splitext(file_path)[0] + suffix

    def _load_cached_mipe(self, cache_path):
        if not os.path.exists(cache_path):
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            return None

        if isinstance(payload, dict):
            if payload.get("config") != self._mipe_config:
                return None
            mipe = payload.get("mipe")
        else:
            mipe = payload

        if not isinstance(mipe, torch.Tensor) or mipe.numel() != self.mipe_scale:
            return None
        return mipe.reshape(self.mipe_scale).float()

    def _save_cached_mipe(self, cache_path, mipe):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        payload = {"config": self._mipe_config, "mipe": mipe.detach().cpu().float()}
        temp_path = f"{cache_path}.{os.getpid()}.tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, cache_path)

    def _load_or_compute_mipe(self, file_path, waveform):
        cache_path = self._mipe_cache_path(file_path)
        mipe = self._load_cached_mipe(cache_path)
        if mipe is not None:
            return mipe

        extractor = self._get_mipe_extractor()
        with torch.no_grad():
            mipe = extractor._batch_mipe(waveform.unsqueeze(0)).squeeze(0).cpu()
        self._save_cached_mipe(cache_path, mipe)
        return mipe.float()

    def precompute_mipe_cache(self):
        if not self.use_cached_mipe:
            return
        total = len(self.segment_list)
        for idx, (file_path, _) in enumerate(self.segment_list, start=1):
            signal = self._prepare_supervised_waveform(file_path)
            waveform = torch.tensor(signal, dtype=torch.float)
            self._load_or_compute_mipe(file_path, waveform)
            if idx % 100 == 0 or idx == total:
                print(f"MIPE cache ready: {idx}/{total}")

    def __getitem__(self, idx):
        """读取 wav 切片，检查采样率，按需加测试噪声并组装 batch item。"""
        file_path, label = self.segment_list[idx]
        
        try:
            sample_rate, signal = wavfile.read(file_path)
        except Exception as e:
            raise RuntimeError(f"🚨 读取音频文件失败: {file_path}. 详情: {e}")

        if sample_rate != self.target_sr:
            raise ValueError(f"🚨 采样率异常！期望 {self.target_sr}Hz, 但文件是 {sample_rate}Hz。")
        
        if len(signal) == 0:
            raise ValueError(f"🚨 发现空音频文件: {file_path}")

        signal = signal.astype(np.float32)
        
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        if getattr(self, 'is_ssl', False):
            gain1 = random.uniform(0.7, 1.3)
            sig_v1 = signal.copy() * gain1
            sig_v1 = add_awgn(sig_v1, snr_db=random.uniform(15, 30), seed_string=None) 
            
            gain2 = random.uniform(0.7, 1.3)
            sig_v2 = signal.copy() * gain2
            sig_v2 = add_awgn(sig_v2, snr_db=random.uniform(10, 25), seed_string=None)
            
            if self.normalize_waveform:
                max_v1 = np.max(np.abs(sig_v1))
                if max_v1 > 0: sig_v1 = sig_v1 / max_v1
                max_v2 = np.max(np.abs(sig_v2))
                if max_v2 > 0: sig_v2 = sig_v2 / max_v2
                
            return (torch.tensor(sig_v1, dtype=torch.float), torch.tensor(sig_v2, dtype=torch.float)), torch.tensor(-1, dtype=torch.long)

        else:
            signal = add_awgn(signal, self.snr_db, seed_string=file_path)

            if self.normalize_waveform:
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val

            waveform = torch.tensor(signal, dtype=torch.float)
            target = torch.tensor(label, dtype=torch.long)

            if self.use_cached_mipe:
                mipe = self._load_or_compute_mipe(file_path, waveform)
                item = {"waveform": waveform, "mipe": mipe}
                if self.return_path:
                    item["path"] = file_path
                return item, target

            if self.return_path:
                return {"waveform": waveform, "path": file_path}, target

            return waveform, target


class ShipsEarDataModule(L.LightningDataModule):
    """LightningDataModule，负责 split 复用/生成、审计和 DataLoader 构建。

    split_file 的 metadata 会记录数据集、协议、比例、随机种子、父目录、
    类别映射、切片长度和采样率；任一关键项不匹配都会自动重新生成 split。
    """
    def __init__(self, parent_folder='./Datasets/ShipsEar', batch_size=None, num_workers=8,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42, 
                 normalize_waveform=False, split_file='shipsear_data_split.json', audit_file='split_audit_report.json',
                 test_snr=None, is_ssl=False, split_protocol='frame_level', segment_length=5,
                 target_sr=16000, dataset_name=None):
        super().__init__()
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "🚨 比例之和必须等于 1.0"
        assert split_protocol in ['frame_level', 'recording_level'], "🚨 split_protocol 必须是 'frame_level' 或 'recording_level'"
        
        self.batch_size = batch_size or {'train': 64, 'val': 64, 'test': 64}
        self.parent_folder = parent_folder
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.normalize_waveform = normalize_waveform
        self.split_file = split_file
        self.audit_file = audit_file
        self.test_snr = test_snr 
        self.is_ssl = is_ssl
        self.split_protocol = split_protocol
        self.segment_length = segment_length
        self.target_sr = target_sr
        self.dataset_name = dataset_name or os.path.basename(os.path.normpath(parent_folder)) or "Unknown"
        self.segment_lists = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _metadata_parent_folder(self):
        return os.path.normpath(os.path.abspath(self.parent_folder))

    def _split_metadata(self, class_mapping):
        return {
            "dataset": self.dataset_name,
            "protocol": self.split_protocol,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "parent_folder": self._metadata_parent_folder(),
            "class_mapping": class_mapping,
            "segment_length": self.segment_length,
            "sample_rate": self.target_sr,
        }

    def _metadata_matches(self, old_meta, expected_meta):
        for key, expected in expected_meta.items():
            actual = old_meta.get(key)
            if isinstance(expected, float):
                if actual is None or abs(float(actual) - expected) > 1e-8:
                    return False, key, actual, expected
            elif actual != expected:
                return False, key, actual, expected
        return True, None, None, None

    def _verify_and_load_split(self, current_class_mapping):
        """校验 split_file metadata，匹配时复用，不匹配时返回 None 触发重建。"""
        if not os.path.exists(self.split_file):
            return None
            
        try:
            with open(self.split_file, 'r') as f:
                data = json.load(f)
                
            meta = data.get('metadata', {})
            expected_meta = self._split_metadata(current_class_mapping)
            matches, key, actual, expected = self._metadata_matches(meta, expected_meta)
            if not matches:
                print(f"Split metadata mismatch on {key}: old={actual}, expected={expected}. Regenerating split.")
                return None

            segment_lists = data.get('segment_lists')
            if not segment_lists or any(split not in segment_lists for split in ['train', 'val', 'test']):
                print("Split file is missing train/val/test lists. Regenerating split.")
                return None

            print(f"Split metadata verified ({meta.get('timestamp')}) [protocol: {self.split_protocol}]")
            return segment_lists
            
        except Exception as e:
            print(f"⚠️ 解析切分文件失败 ({e})，将重新生成...")
            return None

    def save_splits(self, segment_lists, class_mapping):
        """保存 split 列表及其 metadata，保证后续实验可复现和可审计。"""
        metadata = self._split_metadata(class_mapping)
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["shuffle"] = True
        data = {
            "metadata": metadata,
            "segment_lists": segment_lists
        }
        with open(self.split_file, 'w') as f:
            json.dump(data, f, indent=4)

    def _recording_id_from_path(self, file_path):
        return os.path.normpath(os.path.dirname(os.path.abspath(file_path)))

    def _recording_overlap_summary(self, segment_lists):
        """统计 train/val/test 三者之间的录音级 overlap。"""
        recording_sets = {}
        for split in ['train', 'val', 'test']:
            recording_sets[split] = {
                self._recording_id_from_path(file_path)
                for file_path, _ in segment_lists[split]
            }

        overlaps = {}
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            overlap = sorted(recording_sets[left].intersection(recording_sets[right]))
            overlaps[f'{left}-{right}'] = {
                "count": len(overlap),
                "recordings": overlap,
            }
        return recording_sets, overlaps

    def check_segment_leakage(self, segment_lists):
        splits = ['train', 'val', 'test']
        segments = {split: set() for split in splits}

        for split in splits:
            for file_path, _ in segment_lists[split]:
                segments[split].add(os.path.abspath(file_path))

        if segments['train'].intersection(segments['val']) or \
           segments['train'].intersection(segments['test']) or \
           segments['val'].intersection(segments['test']):
            raise ValueError("🚨 严重数据泄漏！Train/Val/Test 之间存在相同的物理切片文件被重复分配！")

    def _print_and_verify_distributions(self, segment_lists, inverse_class_mapping, class_mapping):
        splits = ['train', 'val', 'test']
        num_classes = len(inverse_class_mapping)
        
        protocol_name = "💥 帧级随机划分 (Frame-level Split)" if self.split_protocol == 'frame_level' else "🛡️ 录音级严格划分 (Recording-level Split)"
        
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "class_mapping": class_mapping,
            "protocol": self.split_protocol,
            "splits": {}
        }
        
        print("\n" + "="*75)
        print(f"📊 数据集全局划分与类分布审计报告 [{protocol_name}]")
        print("="*75)
        print(f"🗺️  类别映射: {class_mapping}")
        print("-" * 75)

        for split in splits:
            print(f"[{split.upper():^5} SET]")
            seg_counts = collections.Counter([label for _, label in segment_lists[split]])
            total_segs = sum(seg_counts.values())
            
            audit_data["splits"][split] = {
                "total_segments": total_segs,
                "class_distribution": {}
            }
            
            print(f"总计 -> Segments: {total_segs}")
            
            for class_idx in range(num_classes):
                c_name = inverse_class_mapping[class_idx]
                c_seg = seg_counts.get(class_idx, 0)
                
                if c_seg == 0:
                    raise AssertionError(f"🚨 数据失衡: {split.upper()} 集中, 类别 [{c_name}] 数量为 0！")

                seg_pct = (c_seg / total_segs) * 100 if total_segs > 0 else 0
                audit_data["splits"][split]["class_distribution"][c_name] = {"segments": c_seg}
                
                print(f" Class {c_name:<15} (ID:{class_idx}) | Segs: {c_seg:>4} ({seg_pct:>5.1f}%)")
            print("-" * 75)
            
        with open(self.audit_file, 'w') as f:
            json.dump(audit_data, f, indent=4)
        print(f"💾 审计报告已保存至: {self.audit_file}")
        print("="*75 + "\n")

    def _audit_recording_overlap(self, segment_lists, inverse_class_mapping, class_mapping, stage=None):
        """输出并保存 recording-level 审计报告，overlap 非 0 时停止训练。"""
        recording_sets, recording_overlaps = self._recording_overlap_summary(segment_lists)
        splits = ['train', 'val', 'test']

        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "class_mapping": class_mapping,
            "protocol": self.split_protocol,
            "stage": stage,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "parent_folder": self._metadata_parent_folder(),
            "segment_length": self.segment_length,
            "sample_rate": self.target_sr,
            "splits": {},
            "recording_overlap": recording_overlaps,
        }

        for split in splits:
            seg_counts = collections.Counter([label for _, label in segment_lists[split]])
            audit_data["splits"][split] = {
                "total_segments": sum(seg_counts.values()),
                "total_recordings": len(recording_sets[split]),
                "class_distribution": {},
            }
            for class_idx, class_name in inverse_class_mapping.items():
                class_recordings = {
                    self._recording_id_from_path(file_path)
                    for file_path, label in segment_lists[split]
                    if label == class_idx
                }
                audit_data["splits"][split]["class_distribution"][class_name] = {
                    "segments": seg_counts.get(class_idx, 0),
                    "recordings": len(class_recordings),
                }

        print("\nRecording-level split audit")
        for split in splits:
            split_info = audit_data["splits"][split]
            print(
                f"{split}: segments={split_info['total_segments']} "
                f"recordings={split_info['total_recordings']} "
                f"class_distribution={split_info['class_distribution']}"
            )

        for pair_name, overlap_data in recording_overlaps.items():
            print(f"{pair_name} recording overlap: {overlap_data['count']}")

        with open(self.audit_file, 'w') as f:
            json.dump(audit_data, f, indent=4)

        if self.split_protocol == 'recording_level':
            bad_overlaps = {k: v["count"] for k, v in recording_overlaps.items() if v["count"] > 0}
            if bad_overlaps:
                raise ValueError(f"Recording-level overlap detected: {bad_overlaps}")

    def setup(self, stage=None):
        """准备数据集。

        frame-level 会直接按切片划分；recording-level 先划分录音文件夹，
        再展开为该录音下的所有切片，从源头避免同录音跨集合。
        """
        ships_classes = sorted([f.name for f in os.scandir(self.parent_folder) if f.is_dir()])
        class_mapping = {ship: idx for idx, ship in enumerate(ships_classes)}
        inverse_class_mapping = {idx: ship for ship, idx in class_mapping.items()}

        segment_lists = self._verify_and_load_split(class_mapping)

        # =========================================================
        # 🌟 核心修改区：支持双协议划分逻辑
        # =========================================================
        if segment_lists is None:
            protocol_name = "💥 帧级随机划分 (Frame-level)" if self.split_protocol == 'frame_level' else "🛡️ 录音级严格划分 (Recording-level)"
            print(f"🚀 正在基于比例 {self.train_ratio}:{self.val_ratio}:{self.test_ratio} 生成 [{protocol_name}]...")
            
            # ================= [协议 A] 帧级随机划分 =================
            if self.split_protocol == 'frame_level':
                all_segments_paths = []
                all_segments_labels = []

                for label in ships_classes:
                    label_path = os.path.join(self.parent_folder, label)
                    subfolders = sorted([f.name for f in os.scandir(label_path) if f.is_dir()])
                    
                    for subfolder in subfolders:
                        folder_path = os.path.join(label_path, subfolder)
                        for file in sorted(os.listdir(folder_path)):
                            if file.endswith('.wav'):
                                file_path = os.path.join(folder_path, file)
                                if os.path.isfile(file_path):
                                    all_segments_paths.append(file_path)
                                    all_segments_labels.append(class_mapping[label])
                
                paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
                    all_segments_paths, all_segments_labels, 
                    test_size=self.test_ratio, random_state=self.random_seed, stratify=all_segments_labels
                )

                relative_val_ratio = self.val_ratio / (self.val_ratio + self.train_ratio)
                paths_train, paths_val, labels_train, labels_val = train_test_split(
                    paths_train_val, labels_train_val, 
                    test_size=relative_val_ratio, random_state=self.random_seed, stratify=labels_train_val
                )

                segment_lists = {'train': list(zip(paths_train, labels_train)), 
                                 'val': list(zip(paths_val, labels_val)), 
                                 'test': list(zip(paths_test, labels_test))}

            # ================= [协议 B] 录音级严格划分 =================
            elif self.split_protocol == 'recording_level':
                all_recordings_paths = []
                all_recordings_labels = []

                for label in ships_classes:
                    label_path = os.path.join(self.parent_folder, label)
                    subfolders = sorted([f.name for f in os.scandir(label_path) if f.is_dir()])
                    
                    for subfolder in subfolders:
                        folder_path = os.path.join(label_path, subfolder)
                        if any(f.endswith('.wav') for f in os.listdir(folder_path)):
                            all_recordings_paths.append(folder_path)
                            all_recordings_labels.append(class_mapping[label])
                
                paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
                    all_recordings_paths, all_recordings_labels, 
                    test_size=self.test_ratio, random_state=self.random_seed, stratify=all_recordings_labels
                )

                relative_val_ratio = self.val_ratio / (self.val_ratio + self.train_ratio)
                paths_train, paths_val, labels_train, labels_val = train_test_split(
                    paths_train_val, labels_train_val, 
                    test_size=relative_val_ratio, random_state=self.random_seed, stratify=labels_train_val
                )

                def expand_to_segments(folder_paths, labels):
                    segments = []
                    for folder_path, label in zip(folder_paths, labels):
                        for file in sorted(os.listdir(folder_path)):
                            if file.endswith('.wav'):
                                segments.append((os.path.join(folder_path, file), label))
                    return segments

                segment_lists = {
                    'train': expand_to_segments(paths_train, labels_train),
                    'val': expand_to_segments(paths_val, labels_val),
                    'test': expand_to_segments(paths_test, labels_test)
                }

            # 统一保存切分结果
            self.save_splits(segment_lists, class_mapping)

        # 检查是否发生数据交叉（物理级别的防御）
        self.check_segment_leakage(segment_lists)
        self.segment_lists = segment_lists

        mipe_dataset_kwargs = {
            "target_sr": getattr(self, "target_sr", 16000),
            "use_cached_mipe": getattr(self, "use_cached_mipe", False),
            "mipe_cache_dir": getattr(self, "mipe_cache_dir", None),
            "mipe_cache_root": self.parent_folder,
            "mipe_m": getattr(self, "mipe_m", 3),
            "mipe_tau": getattr(self, "mipe_tau", 1),
            "mipe_c": getattr(self, "mipe_c", 10),
            "mipe_scale": getattr(self, "mipe_scale", 10),
            "disable_mipe": getattr(self, "disable_mipe", False),
        }

        self.train_dataset = ShipsEarDataset(
            segment_lists['train'],
            normalize_waveform=self.normalize_waveform,
            snr_db=None,
            is_ssl=self.is_ssl,
            **mipe_dataset_kwargs
        )
        self.val_dataset = ShipsEarDataset(
            segment_lists['val'],
            normalize_waveform=self.normalize_waveform,
            snr_db=None,
            is_ssl=False,
            **mipe_dataset_kwargs
        )
        self.test_dataset = ShipsEarDataset(
            segment_lists['test'],
            normalize_waveform=self.normalize_waveform,
            snr_db=self.test_snr,
            is_ssl=False,
            return_path=True,
            **mipe_dataset_kwargs
        )

        if getattr(self, "precompute_mipe", False) and getattr(self, "use_cached_mipe", False):
            print("Precomputing MIPE cache for train/val/test splits...")
            self.train_dataset.precompute_mipe_cache()
            self.val_dataset.precompute_mipe_cache()
            self.test_dataset.precompute_mipe_cache()

        self._print_and_verify_distributions(segment_lists, inverse_class_mapping, class_mapping)
        self._audit_recording_overlap(segment_lists, inverse_class_mapping, class_mapping, stage=stage)
        
        if self.test_snr is not None:
            print(f"🌪️  [高能预警] 当前测试集已注入固定随机种子的 AWGN 白噪声，信噪比 SNR = {self.test_snr} dB")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], num_workers=self.num_workers, pin_memory=True)
