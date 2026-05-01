import os
import librosa
import soundfile as sf
import math

# 🌟 修改点 1: 默认 dest_dir 改为 'DeepShip_Segments_3s/'，segment_length 改为 3
def Generate_Segments_DeepShip(src_dir='DeepShip/', dest_dir='DeepShip_Segments_3s/', target_sr=16000, segment_length=3):
    # 如果目标文件夹已经存在，说明切过了，直接返回
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"✅ 发现已存在的 3s 切片目录 {dest_dir}，跳过预处理！可以直接训练！")
        return dest_dir

    print(f"🚀 开始将 {src_dir} 的原始音频切片并重采样至 {dest_dir} (这需要几分钟)...")
    os.makedirs(dest_dir, exist_ok=True)
    
    # 获取类别 (Cargo, Passengership, Tanker, Tug)
    ship_classes = sorted([f.name for f in os.scandir(src_dir) if f.is_dir()])
    total_processed = 0

    for ship_class in ship_classes:
        class_src_dir = os.path.join(src_dir, ship_class)
        class_dest_dir = os.path.join(dest_dir, ship_class)
        os.makedirs(class_dest_dir, exist_ok=True)

        # DeepShip 特有结构: Class / Vessel_Name / audio.wav
        vessels = sorted([f.name for f in os.scandir(class_src_dir) if f.is_dir()])
        
        for vessel in vessels:
            vessel_dir = os.path.join(class_src_dir, vessel)
            wav_files = [f for f in os.listdir(vessel_dir) if f.lower().endswith('.wav')]
            
            for file_name in wav_files:
                file_path = os.path.join(vessel_dir, file_name)
                
                # 为该音频创建一个专属的切片存放子文件夹
                subfolder_name = f"{vessel}_{os.path.splitext(file_name)[0]}"
                segment_folder_path = os.path.join(class_dest_dir, subfolder_name)
                os.makedirs(segment_folder_path, exist_ok=True)
                
                print(f"正在处理 (3s切片): [{ship_class}] {vessel} -> {file_name}")
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    
                    # 强行降采样到 16000Hz
                    if sr != target_sr:
                        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

                    duration = len(audio)
                    # 🌟 修改点 2: 这里使用的是传入的 segment_length (现在是 3)
                    segment_duration = target_sr * segment_length 
                    number_of_segments = math.ceil(duration / segment_duration)

                    for i in range(number_of_segments):
                        start_i = i * segment_duration
                        end_i = start_i + segment_duration

                        if end_i > duration:
                            end_i = duration

                        output_segment = audio[start_i:end_i]

                        # 严格只保留完整的 3 秒切片
                        if end_i - start_i == segment_duration:
                            segment_file_path = os.path.join(segment_folder_path, f'Seg_{i+1}.wav')
                            sf.write(segment_file_path, output_segment, samplerate=target_sr)
                    total_processed += 1
                except Exception as e:
                    print(f"❌ 处理文件 {file_path} 时出错: {e}")

    print(f"\n🎉 完美！成功处理并切片了 {total_processed} 个原始音频文件 (3秒版本)。")
    return dest_dir

if __name__ == '__main__':
    # 单独运行时的默认路径也对应更新
    Generate_Segments_DeepShip(src_dir='../DeepShip/', dest_dir='../DeepShip_Segments_3s/')

# python -c "from Datasets.DeepShip_Data_Preprocessing import Generate_Segments_DeepShip; Generate_Segments_DeepShip(src_dir='DeepShip/', dest_dir='DeepShip_Segments_3s/', target_sr=16000, segment_length=3)"
    