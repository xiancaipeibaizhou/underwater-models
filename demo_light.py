"""训练入口脚本。

本文件负责把命令行参数整理为 Params，完成数据预处理、DataModule
构建、LitModel 构建、复杂度统计、训练/验证/测试与结果落盘。

关键参数：
- data_selection: 0=DeepShip，1=ShipsEar。
- split_protocol: frame_level 或 recording_level；正式实验使用 recording_level。
- train_ratio / val_ratio / test_ratio: 数据划分比例，和必须为 1。
- test_only / ckpt_path: 跳过训练，直接加载指定 checkpoint 做测试。
"""

import numpy as np
import argparse
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback  # 🌟 引入基类
import os
import csv  # 🌟 引入 CSV 写入模块
import json
import shutil
import copy
import time
from Datasets.DeepShip_Data_Preprocessing import Generate_Segments_DeepShip
from Utils.LitModel import LitModel

from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_big_number(value):
    if value is None:
        return "unknown"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)

def count_macs(model, input_shape):
    try:
        from thop import profile

        training_state = model.training
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(input_shape)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        model.train(training_state)
        return int(macs), "thop"
    except Exception:
        pass

    hooks = []
    macs = {"total": 0}

    def conv_hook(module, inputs, output):
        x = inputs[0]
        out = output
        if not isinstance(out, torch.Tensor) or out.ndim < 4:
            return
        batch_size = int(x.shape[0])
        out_channels = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups))
        macs["total"] += batch_size * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if not isinstance(out, torch.Tensor):
            return
        macs["total"] += int(out.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(input_shape)
        model(dummy)
    model.train(training_state)

    for hook in hooks:
        hook.remove()
    return int(macs["total"]), "conv_linear_hooks"

@torch.no_grad()
def benchmark_latency(model, input_shape, device, warmup=30, repeats=100):
    model = model.to(device)
    model.eval()
    x = torch.randn(input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats

def save_model_complexity(model_wrapper, Params, save_dir):
    """统计当前模型主体的参数量、MACs 与 batch=1 推理延迟。

    这里统计的是 `model_wrapper.model`，即 Log-Mel 特征提取层之后的分类网络。
    因此 UATR_KNN、ShuffleFAC、FA_UATR_KNN 会使用同一套复杂度统计逻辑。
    """
    n_mels = Params.get('number_mels', 128)
    sample_rate = Params.get('sample_rate', 16000)
    segment_length = Params.get('segment_length', 5)
    hop_length = Params.get('hop_length', 512)
    t_dim = int((sample_rate * segment_length) / hop_length) + 1
    if Params.get('Model_name') == 'ShuffleFAC_CLIPGRAPH':
        input_shape = (1, Params.get('clips_per_recording', 4), 1, n_mels, t_dim)
    else:
        input_shape = (1, 1, n_mels, t_dim)

    target_model = model_wrapper.model
    total_params = sum(p.numel() for p in target_model.parameters())
    trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)

    mac_model = copy.deepcopy(target_model).cpu()
    macs, macs_method = count_macs(mac_model, input_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latency_model = copy.deepcopy(target_model)
    latency_s = benchmark_latency(
        latency_model,
        input_shape,
        device,
        warmup=Params.get('latency_warmup', 30),
        repeats=Params.get('latency_repeats', 100),
    )

    complexity = {
        "input_shape": list(input_shape),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "macs": int(macs),
        "macs_formatted": format_big_number(int(macs)),
        "macs_method": macs_method,
        "latency_seconds_batch1": float(latency_s),
        "latency_ms_batch1": float(latency_s * 1000.0),
        "latency_device": str(device),
    }

    with open(os.path.join(save_dir, "model_complexity.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input shape: {complexity['input_shape']}\n")
        f.write(f"Total params: {complexity['total_params']}\n")
        f.write(f"Trainable params: {complexity['trainable_params']}\n")
        f.write(f"MACs: {complexity['macs']}\n")
        f.write(f"MACs formatted: {complexity['macs_formatted']}\n")
        f.write(f"MACs method: {complexity['macs_method']}\n")
        f.write(f"Latency seconds/batch1: {complexity['latency_seconds_batch1']:.9f}\n")
        f.write(f"Latency ms/batch1: {complexity['latency_ms_batch1']:.6f}\n")
        f.write(f"Latency device: {complexity['latency_device']}\n")

    return complexity

def configure_mipe_cache(data_module, Params):
    """把 MIPE 缓存相关参数挂到 DataModule。

    该函数只服务于保留的多特征/辅助实验路线。Log-Mel 主线模型
    如 UATR_KNN、ShuffleFAC、FA_UATR_KNN 不依赖 MIPE 输入。
    """
    data_module.target_sr = Params.get('sample_rate', 16000)
    data_module.use_cached_mipe = Params.get('use_cached_mipe', False)
    data_module.precompute_mipe = Params.get('precompute_mipe', False)
    data_module.mipe_cache_dir = Params.get('mipe_cache_dir')
    data_module.mipe_m = Params.get('mipe_m', 3)
    data_module.mipe_tau = Params.get('mipe_tau', 1)
    data_module.mipe_c = Params.get('mipe_c', 10)
    data_module.mipe_scale = Params.get('mipe_scale', 10)
    data_module.disable_mipe = Params.get('disable_mipe', False)

# ==========================================
# 🌟 核心暴力破解：绝对不依赖官方 Logger，直接去内存里抠数据！
# ==========================================
class ForceMetricsWriter(Callback):
    def __init__(self, save_dir):
        self.filepath = os.path.join(save_dir, 'epoch_metrics.csv')
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # 跳过一开始的 Sanity Check
        if trainer.sanity_checking:
            return
            
        # 读取 EarlyStopping 正在监控的内存字典
        metrics = trainer.callback_metrics
        row = {'epoch': trainer.current_epoch}
        for k, v in metrics.items():
            row[k] = v.item() if isinstance(v, torch.Tensor) else v
            
        self.history.append(row)
        
        # 动态收集所有出现过的键名 (解决 Epoch 1 突然冒出 train_loss 的问题)
        all_keys = set()
        for r in self.history:
            all_keys.update(r.keys())
        
        # 让 'epoch' 永远排在第一列，其他按字母排序
        sorted_keys = ['epoch'] + sorted([k for k in all_keys if k != 'epoch'])

        # 每次覆盖写入，绝不缓冲！
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerows(self.history)


def main(Params):
    """根据 Params 启动一次或多次实验。

    该函数不直接解释命令行，而是消费 parse_args 后整理出的 Params。
    数据集选择、split 协议、模型名称和训练超参数都在这里汇合。
    """
    model_name = Params['Model_name']
    if model_name == 'ShuffleFAC_CLIPGRAPH':
        Params['recording_bag_mode'] = True
    batch_size = Params['batch_size']
    num_workers = Params['num_workers']
    
    if Params['data_selection'] == 1:
        src_dataset_dir = 'shipsEar_AUDIOS/'
        seg_len = Params.get('segment_length', 5)
        dataset_dir = src_dataset_dir if seg_len == 5 else f"shipsEar_AUDIOS_{seg_len}s/"
        dataset_dir = Generate_Segments(
            src_dataset_dir,
            target_sr=Params.get('sample_rate', 16000),
            segment_length=seg_len,
            dest_dir=None if seg_len == 5 else dataset_dir,
        )
        data_module = ShipsEarDataModule(
            parent_folder=dataset_dir, 
            batch_size=batch_size, 
            num_workers=num_workers,
            dataset_name="ShipsEar",
            target_sr=Params.get('sample_rate', 16000),
            test_snr=Params.get('test_snr'),
            train_ratio=Params.get('train_ratio', 0.6),
            val_ratio=Params.get('val_ratio', 0.2),
            test_ratio=Params.get('test_ratio', 0.2),
            segment_length=Params.get('segment_length', 5),
            split_file='shipsear_data_split.json', 
            audit_file='split_audit_report.json',
            split_protocol=Params.get('split_protocol', 'recording_level'),
            random_seed=Params.get('split_seed', 42),
            recording_bag_mode=Params.get('recording_bag_mode', False),
            clips_per_recording=Params.get('clips_per_recording', 4)
        )
        configure_mipe_cache(data_module, Params)
        num_classes = 5 
        DataName = "ShipsEar"

    elif Params['data_selection'] == 0:  
        src_dir = 'DeepShip/' 
        
        # 🌟 核心修复：根据切片长度动态命名输出文件夹，彻底解决缓存冲突！
        
        seg_len = Params.get('segment_length', 5)
        # 🌟 强行修复：如果是 5 秒，直接用回你原本那个旧文件夹，绝不重新切！
        dest_dir = 'DeepShip_Segments/' if seg_len == 5 else f'DeepShip_Segments_{seg_len}s/'
        
        dataset_dir = Generate_Segments_DeepShip(
            src_dir=src_dir,
            dest_dir=dest_dir,
            target_sr=Params.get('sample_rate', 16000),
            segment_length=seg_len
        )
        
        num_classes = len([f.name for f in os.scandir(dataset_dir) if f.is_dir()])
        
        data_module = ShipsEarDataModule(
            parent_folder=dataset_dir, 
            batch_size=batch_size, 
            num_workers=num_workers,
            dataset_name="DeepShip",
            target_sr=Params.get('sample_rate', 16000),
            test_snr=Params.get('test_snr'),
            train_ratio=Params.get('train_ratio', 0.6),
            val_ratio=Params.get('val_ratio', 0.2),
            test_ratio=Params.get('test_ratio', 0.2),
            segment_length=Params.get('segment_length', 5),
            split_file='deepship_data_split.json', 
            audit_file='deepship_audit_report.json',
            split_protocol=Params.get('split_protocol', 'recording_level'),
            random_seed=Params.get('split_seed', 42),
            recording_bag_mode=Params.get('recording_bag_mode', False),
            clips_per_recording=Params.get('clips_per_recording', 4)
        )
        configure_mipe_cache(data_module, Params)
        DataName = "DeepShip"
    else:
        raise ValueError('不支持的数据集选择，仅支持 0=DeepShip, 1=ShipsEar')

    torch.set_float32_matmul_precision('medium') 
    
    snr_suffix = f"_SNR{Params['test_snr']}" if Params.get('test_snr') is not None else "_Clean"
    group_str = f"G{int(Params['use_graph'])}_P{int(Params['use_prior_mask'])}_TE{int(Params['use_temporal_encoder'])}_TA{int(Params['use_temporal_attention'])}{snr_suffix}"
    
    print(f'\n🚀 Starting Experiments for {DataName} using {model_name} | Group: {group_str} 🚀')
    
    csv_file = "htan_ablations_results.csv" 
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Model,ExpTime,Group,Run_Index,Seed,ACC,APR_Weighted,RE_Weighted,F1_Macro,F1_Weighted,Val_Macro_F1\n")
    
    numRuns = 1 if Params.get('test_only') else Params.get('num_runs', 3)
    start_run_index = Params.get('run_index', 0)
    if Params.get('test_only'):
        run_sequence = [start_run_index]
    else:
        run_sequence = range(start_run_index, start_run_index + numRuns)

    for run_number in run_sequence:
        model_seed_base = Params.get('model_seed', 42)
        current_seed = model_seed_base + run_number
        seed_everything(current_seed, workers=True)
        
        exp_folder_name = f"{DataName}_{Params['exp_time']}" if Params.get('exp_time') else DataName
        save_dir = os.path.join("results", exp_folder_name, group_str, f"Run_{run_number}")
        os.makedirs(save_dir, exist_ok=True)

        model_config = {
            # model_config.json 是每个 run 的实验快照，服务器正式实验后
            # 依靠它确认模型、split 协议、数据参数和复杂度统计是否一致。
            "model": model_name,
            "uatr_variant": Params.get('uatr_variant'),
            "run_index": run_number,
            "training_seed": current_seed,
            "model_seed_base": model_seed_base,
            "split_random_seed": data_module.random_seed,
            "split_seed": Params.get('split_seed', 42),
            "segment_length": Params.get('segment_length', 5),
            "split_protocol": Params.get('split_protocol', 'recording_level'),
            "train_ratio": Params.get('train_ratio', 0.6),
            "val_ratio": Params.get('val_ratio', 0.2),
            "test_ratio": Params.get('test_ratio', 0.2),
            "sample_rate": Params.get('sample_rate', 16000),
            "window_length": Params.get('window_length', 2048),
            "hop_length": Params.get('hop_length', 512),
            "number_mels": Params.get('number_mels', 128),
            "n_mfcc": Params.get('n_mfcc', 20),
            "stft_bins": Params.get('stft_bins', 64),
            "aux_target_dim": Params.get('aux_target_dim', 208),
            "aux_hidden_dim": Params.get('aux_hidden_dim'),
            "aux_loss_weight": Params.get('aux_loss_weight', 0.05),
            "mipe_m": Params.get('mipe_m', 3),
            "mipe_tau": Params.get('mipe_tau', 1),
            "mipe_c": Params.get('mipe_c', 10),
            "mipe_scale": Params.get('mipe_scale', 10),
            "disable_mipe": Params.get('disable_mipe', False),
            "use_cached_mipe": Params.get('use_cached_mipe', False),
            "precompute_mipe": Params.get('precompute_mipe', False),
            "mipe_cache_dir": Params.get('mipe_cache_dir'),
            "fusion_dim": Params.get('fusion_dim', 128),
            "fusion_dropout": Params.get('fusion_dropout', 0.2),
            "dropout": Params.get('dropout', 0.2),
            "knn_k": Params.get('knn_k', 8),
            "uatr_depth": Params.get('uatr_depth', 1),
            "fa_target_freq": Params.get('fa_target_freq', 4),
            "fa_arch": Params.get('fa_arch', 'parallel'),
            "pos_type": Params.get('pos_type', '2d'),
            "knn_metric": Params.get('knn_metric', 'cosine'),
            "knn_source": Params.get('knn_source', 'pre_trans'),
            "gate_type": Params.get('gate_type', 'token'),
            "gate_init_bias": Params.get('gate_init_bias', -2.0),
            "fusion_mode": Params.get('fusion_mode', 'gated'),
            "recording_bag_mode": Params.get('recording_bag_mode', False),
            "clips_per_recording": Params.get('clips_per_recording', 4),
            "shufflefac_gamma": Params.get('shufflefac_gamma', 16),
            "graph_hidden_dim": Params.get('graph_hidden_dim', 128),
            "graph_layers": Params.get('graph_layers', 1),
            "graph_k": Params.get('graph_k', 2),
            "edge_mode": Params.get('edge_mode', 'temporal_similarity'),
            "graph_pooling": Params.get('graph_pooling', 'attention'),
        }
        with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        data_module.audit_file = os.path.join(save_dir, "split_audit_pre.txt")
        data_module.setup(stage="pre_train_audit")
        shutil.copyfile(data_module.audit_file, os.path.join(save_dir, "split_audit.txt"))
        data_module.audit_file = os.path.join(save_dir, "split_audit_runtime.txt")
        
        print(f'\n>>> Starting [ {group_str} ] | Run {run_number+1}/{numRuns} (Seed: {current_seed}) <<<\n')
        print(f'📁 Outputs will be saved strictly to: {save_dir}\n')
    
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename='best_model',
            monitor='val_macro_f1', 
            save_top_k=1,
            mode='max',             
            verbose=False,
            save_weights_only=True
        )
        # best checkpoint 只按验证集 Macro-F1 选择，测试集只在训练结束后评估一次，
        # 避免把测试集反馈到模型选择中。
    
        early_stopping_callback = EarlyStopping(
            monitor='val_macro_f1', 
            patience=Params['patience'],
            verbose=True,
            mode='max'              
        )

        # 🌟 实例化我们写的暴力写入插件
        force_metrics_writer = ForceMetricsWriter(save_dir=save_dir)

        model_wrapper = LitModel(Params, model_name, num_classes)
        complexity = save_model_complexity(model_wrapper, Params, save_dir)
        model_config["complexity"] = complexity
        with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        if run_number == 0:
            num_params = complexity["trainable_params"]
            print(f'\n💡 Total Trainable Parameters: {num_params / 1e6:.4f} M')
            print(f"💡 MACs: {complexity['macs_formatted']} | Latency: {complexity['latency_ms_batch1']:.3f} ms/batch1\n")

        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback, force_metrics_writer],
            deterministic=False,
            logger=False, 
            log_every_n_steps=10,
            enable_progress_bar=True,
            accelerator='gpu',       
            devices="auto"
        )
        
        if not Params.get('test_only'):
            trainer.fit(model=model_wrapper, datamodule=data_module) 
            best_val_macro_f1 = checkpoint_callback.best_model_score.item() # 🌟 统一使用新名字
            best_model_path = checkpoint_callback.best_model_path
        else:
            print(f"\n⚠️ [鲁棒性测试模式] 跳过训练，直接加载 Clean 权重: {Params['ckpt_path']}")
            if Params.get('ckpt_path') is None or not os.path.exists(Params['ckpt_path']):
                raise FileNotFoundError(f"🚨 找不到指定的权重文件: {Params.get('ckpt_path')}！请检查路径。")
            best_model_path = Params['ckpt_path']
            best_val_macro_f1 = 0.0 # 🌟 这里也要同步改为新名字

        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
        )
    
        best_model.test_save_dir = save_dir
        trainer.test(model=best_model, datamodule=data_module)

        data_module.audit_file = os.path.join(save_dir, "split_audit_post.txt")
        data_module.setup(stage="post_test_audit")
        shutil.copyfile(data_module.audit_file, os.path.join(save_dir, "split_audit.txt"))
        
        metrics = best_model.custom_metrics
    
        with open(csv_file, "a") as f:
            # 🌟 这里的最后一个变量修改为 best_val_macro_f1，解决你的报错
            f.write(f"{DataName},{model_name},{Params.get('exp_time', 'None')},{group_str},{run_number},{current_seed},"
                    f"{metrics['ACC']:.4f},{metrics['APR_Weighted']:.4f},{metrics['RE_Weighted']:.4f},"
                    f"{metrics['F1_Macro']:.4f},{metrics['F1_Weighted']:.4f},{best_val_macro_f1:.4f}\n")
    
        with open(os.path.join(save_dir, "metrics.txt"), "w") as file:
            file.write(f"=== {model_name} ({group_str}) Run {run_number} ===\n")
            file.write(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}\n")
            for k, v in metrics.items():
                file.write(f"Test {k}: {v:.4f}\n")
            file.write(f"Trainable Params: {complexity['trainable_params']}\n")
            file.write(f"MACs: {complexity['macs']}\n")
            file.write(f"MACs formatted: {complexity['macs_formatted']}\n")
            file.write(f"Latency ms/batch1: {complexity['latency_ms_batch1']:.6f}\n")

def parse_args():
    """定义服务器实验常用命令行参数。

    `--model` 选择模型；`--data_selection` 选择数据集；
    `--split_protocol recording_level` 是当前公平对比的正式协议。
    """
    parser = argparse.ArgumentParser(description='Run Advanced UATR HTAN Experiments')
    # 数据集与划分协议。正式对比使用 recording_level，避免同录音切片泄露。
    parser.add_argument('--model', type=str, default='HTAN', help='Select baseline model architecture')
    parser.add_argument('--data_selection', type=int, default=0, help='Dataset selection: 0=DeepShip, 1=ShipsEar')
    parser.add_argument('--split_protocol', type=str, default='recording_level',
                        choices=['frame_level', 'recording_level'])
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed used only for train/val/test split generation')
    parser.add_argument('--model_seed', type=int, default=42,
                        help='Base random seed for model training; actual seed is model_seed + run_index')
    
    parser.add_argument('--use_graph', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_prior_mask', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_temporal_encoder', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_temporal_attention', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    
    parser.add_argument('--train_batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16, help='input batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=16, help='input batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train each model for')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for Dataloader')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank', help='Audio feature for extraction')
    parser.add_argument('--window_length', type=int, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, default=512, help='hop length')
    parser.add_argument('--number_mels', type=int, default=128, help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Dataset Sample Rate')
    parser.add_argument('--segment_length', type=int, default=5, help='Dataset Segment Length')
    parser.add_argument('--n_mfcc', type=int, default=20)
    parser.add_argument('--stft_bins', type=int, default=64)
    parser.add_argument('--aux_target_dim', type=int, default=208)
    parser.add_argument('--aux_hidden_dim', type=int, default=None)
    parser.add_argument('--aux_loss_weight', type=float, default=0.05)
    parser.add_argument('--mipe_m', type=int, default=3)
    parser.add_argument('--mipe_tau', type=int, default=1)
    parser.add_argument('--mipe_c', type=int, default=10)
    parser.add_argument('--mipe_scale', type=int, default=10)
    parser.add_argument('--disable_mipe', action='store_true')
    parser.add_argument('--use_cached_mipe', action='store_true')
    parser.add_argument('--precompute_mipe', action='store_true')
    parser.add_argument('--mipe_cache_dir', type=str, default=None)
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--fusion_dropout', type=float, default=0.2)
    parser.add_argument('--knn_k', type=int, default=8)
    parser.add_argument('--uatr_depth', type=int, default=1)
    parser.add_argument('--fa_target_freq', type=int, default=4, choices=[1, 4, 8])
    parser.add_argument('--fa_arch', type=str, default='parallel', choices=['serial', 'parallel'])
    parser.add_argument('--pos_type', type=str, default='2d', choices=['1d', '2d', 'none'])
    parser.add_argument('--knn_metric', type=str, default='cosine', choices=['l2', 'cosine'])
    parser.add_argument('--knn_source', type=str, default='pre_trans', choices=['pre_trans', 'post_trans'])
    parser.add_argument('--gate_type', type=str, default='token', choices=['scalar', 'token', 'element'])
    parser.add_argument('--gate_init_bias', type=float, default=-2.0)
    parser.add_argument('--fusion_mode', type=str, default='gated',
                        choices=['gated', 'trans_only', 'graph_only', 'fixed'])
    parser.add_argument('--recording_bag_mode', action='store_true',
                        help='Use recording-level bags instead of segment-level samples')
    parser.add_argument('--clips_per_recording', type=int, default=4)
    parser.add_argument('--shufflefac_gamma', type=int, default=16)
    parser.add_argument('--graph_hidden_dim', type=int, default=128)
    parser.add_argument('--graph_layers', type=int, default=1)
    parser.add_argument('--graph_k', type=int, default=2)
    parser.add_argument('--edge_mode', type=str, default='temporal_similarity',
                        choices=['temporal', 'similarity', 'temporal_similarity'])
    parser.add_argument('--graph_pooling', type=str, default='attention',
                        choices=['mean', 'attention'])

    parser.add_argument('--test_snr', type=float, default=None, help='测试集注入的 SNR (dB)')
    # test_only 模式用于服务器上加载已有 ckpt 做鲁棒性/噪声测试，不触发训练。
    parser.add_argument('--test_only', action='store_true', help='跳过训练，仅加载权重进行测试')
    parser.add_argument('--ckpt_path', type=str, default=None, help='仅测试模式下加载的权重路径')             
    parser.add_argument('--exp_time', type=str, default='', help='按时间戳生成独立文件夹防止覆盖')    
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 Regularization weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the model')                   
    parser.add_argument('--run_index', type=int, default=0, help='指定当前跑的是第几个 Seed') 
    parser.add_argument('--num_runs', type=int, default=3, help='Number of independent training runs')
    parser.add_argument('--latency_warmup', type=int, default=30)
    parser.add_argument('--latency_repeats', type=int, default=100)
    parser.add_argument('--uatr_variant', type=str, default='C',
                        choices=['A', 'B', 'C', 'D'],
                        help='A=Patch+Trans, B=Patch+KNN-GNN, C=Patch+Trans+KNN-GNN, D=PatchOnly')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = vars(args).copy()
    
    params['Model_name'] = args.model
    params['batch_size'] = {
        'train': args.train_batch_size,
        'val': args.val_batch_size,
        'test': args.test_batch_size
    }
    
    for key in ['use_graph', 'use_prior_mask', 'use_temporal_encoder', 'use_temporal_attention']:
        params[key] = bool(params[key])
        
    main(params)
