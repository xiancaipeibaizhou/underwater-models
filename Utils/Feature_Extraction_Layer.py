# Feature_Extraction_Layer.py

import torch.nn as nn
import torch
import torchaudio.transforms as T
from .LogMelFilterBank import MelSpectrogramExtractor  

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=4096, 
                 hop_length=512, number_mels=64, segment_length=5, use_spec_aug=False): 
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.sample_frequency = sample_rate 
        self.num_channels = 1
        self.input_feature = input_feature
        
        # Initialize logmelfbank
        win_length = window_length
        n_fft = window_length
        hop_length = hop_length 
        n_mels = number_mels
        fmin = 1
        fmax = 8000
        
        self.LogMelFBank = MelSpectrogramExtractor(
            sample_rate=sample_rate, 
            n_fft=n_fft,
            win_length=win_length, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmin=fmin, 
            fmax=fmax
        )

        self.features = {'LogMelFBank': self.LogMelFBank}
        
        # =====================================================================
        # >>> 核心创新：物理声谱掩码数据增强 (SpecAugment) <<<
        # 强制增加训练难度，逼迫 PhysicalHarmonicGCN 发挥谐波推导与时序重建的作用
        # =====================================================================
        self.use_spec_aug = use_spec_aug
        self.freq_masking = T.FrequencyMasking(freq_mask_param=8) 
        self.time_masking = T.TimeMasking(time_mask_param=15)
        # =====================================================================
                
        self.output_dims = None
        self.calculate_output_dims()

    def calculate_output_dims(self):
        try:
            length_in_seconds = self.segment_length  
            samples = int(self.sample_rate * length_in_seconds)
            dummy_input = torch.randn(1, samples)  
            with torch.no_grad():
                output = self.features[self.input_feature](dummy_input)
                self.output_dims = output.shape
        except Exception as e:
            print(f"Failed to calculate output dimensions: {e}\n")
            self.output_dims = None
            
    # 修改 forward 函数，增加 return_clean 参数
    def forward(self, x, return_clean=False):
        x_clean = self.features[self.input_feature](x) 
        x_masked = x_clean.clone()

        # 🌟 现在由参数严格控制是否开启掩码
        if self.training and self.use_spec_aug:
            x_masked = self.freq_masking(x_masked)
            x_masked = self.time_masking(x_masked)

        x_masked = x_masked.unsqueeze(1) # 增加通道维度 -> [Batch, 1, Freq, Time]
        x_clean = x_clean.unsqueeze(1)   # [Batch, 1, Freq, Time]

        # 🌟 如果处于自监督掩码重建模式，同时返回掩码版和干净版
        if return_clean:
            return x_masked, x_clean
            
        return x_masked