import os
import librosa
import soundfile as sf
import math

def Generate_Segments(dataset_dir, target_sr=16000, segment_length=5):
    '''
    dataset_dir: 包含 A, B, C, D, E 子文件夹的数据集根目录
    target_sr: 目标采样率 (16000 Hz)
    segment_length: 切片长度 (5秒)
    '''
    # 直接使用我们已经分好类的五个文件夹
    ship_type = ['A', 'B', 'C', 'D', 'E']
    
    segmentation_needed = False
    
    # 检查是否需要切片 (判断是否已经存在切片子文件夹)
    for ship in ship_type:
        ship_dir = os.path.join(dataset_dir, ship)
        if not os.path.exists(ship_dir):
            continue
            
        # 找出该类别文件夹下所有的原始 wav 文件 (忽略已生成的文件夹)
        wav_files = [f for f in os.listdir(ship_dir) if f.lower().endswith('.wav') and os.path.isfile(os.path.join(ship_dir, f))]
        
        for file_name in wav_files:
            subfolder_name = os.path.splitext(file_name)[0]
            segment_folder_path = os.path.join(ship_dir, subfolder_name)
            
            if not os.path.exists(segment_folder_path):
                segmentation_needed = True
                break
        if segmentation_needed:
            break

    if segmentation_needed:
        print("开始提取音频特征与切片，这可能需要几分钟时间...")
    else:
        print("所有音频已切片完毕，跳过处理。可以直接去训练啦！")
        return

    # 正式开始切片流程
    total_processed = 0
    for ship in ship_type:
        ship_dir = os.path.join(dataset_dir, ship)
        if not os.path.exists(ship_dir):
            continue
            
        wav_files = [f for f in os.listdir(ship_dir) if f.lower().endswith('.wav') and os.path.isfile(os.path.join(ship_dir, f))]
        
        for file_name in wav_files:
            subfolder_name = os.path.splitext(file_name)[0]
            segment_folder_path = os.path.join(ship_dir, subfolder_name)
            
            if os.path.exists(segment_folder_path):
                continue  # 已经切片过的文件直接跳过

            # 加载原始音频
            file_path = os.path.join(ship_dir, file_name)
            print(f"正在处理类别 [{ship}] -> {file_name}")
            try:
                audio, sr = librosa.load(file_path, sr=None)
                
                # 重采样到目标频率
                audio_resampled = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

                # 切片计算
                duration = len(audio_resampled)
                segment_duration = target_sr * segment_length
                number_of_segments = math.ceil(duration / segment_duration)

                os.makedirs(segment_folder_path, exist_ok=True)

                for i in range(number_of_segments):
                    start_i = i * segment_duration
                    end_i = start_i + segment_duration

                    if end_i > duration:
                        end_i = duration

                    output_music = audio_resampled[start_i:end_i]

                    # 严格防泄露逻辑：只保存完整长度的切片（丢弃最后不足5秒的尾巴）
                    if end_i - start_i == segment_duration:
                        segment_file_path = os.path.join(segment_folder_path, f'{subfolder_name}-Segment_{i+1}.wav')
                        sf.write(segment_file_path, output_music, samplerate=target_sr)
                total_processed += 1
            except Exception as e:
                print(f"读取或处理文件 {file_name} 时出错: {e}")

    print(f"\n完美！成功处理了 {total_processed} 个原始音频文件。现在可以运行 demo_light.py 进行训练了！")

if __name__ == '__main__':
    # 确保这个路径指向包含 A,B,C,D,E 的目录
    dataset_dir = 'shipsEar_AUDIOS/'
    Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)