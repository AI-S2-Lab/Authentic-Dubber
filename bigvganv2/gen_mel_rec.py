import os
import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write
from tqdm import tqdm  # 用于进度条
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BigVGAN 模型
model = bigvgan.BigVGAN.from_pretrained('/sdc/data/zhaoyuan/bigvgan_v2_22khz_80band_fmax8k_256x', use_cuda_kernel=False)

# 移除模型中的权重归一化，并设置为评估模式
model.remove_weight_norm()
model = model.eval().to(device)

# 定义处理函数
def extract(file_path, gt_mel_path, out_path):
    # 确保输出路径存在
    os.makedirs(out_path, exist_ok=True)

    with open(file_path, 'r') as file:
        all_lines = file.readlines()
        num_lines = len(all_lines)

        # 添加进度条
        for i in tqdm(ange(num_lines), desc="Processing"):
            parts = all_lines[i].strip().split('|')  # 按 '|' 分割每行数据

            basename = parts[0]
            speaker = parts[1]

            # 构建 mel 文件路径
            gt_mel_path_i = os.path.join(
                gt_mel_path,
                f"{speaker}-mel-{basename}.npy"
            )

            # 加载 ground truth mel 文件
            gt_mel_npy = np.load(gt_mel_path_i)  # 加载为 numpy 数组
            gt_mel_tensor = torch.FloatTensor(gt_mel_npy).transpose(0, 1).unsqueeze(0).to(device)  # 转换为 PyTorch 张量

            # 使用模型生成音频
            with torch.inference_mode():
                gt_mel_rec = model(gt_mel_tensor).squeeze(0).cpu()
            
            # 转换为 16-bit PCM 格式
            gt_mel_rec = torch.clamp(gt_mel_rec, -1.0, 1.0)  # 限制范围 [-1, 1]
            gt_mel_rec_int16 = (gt_mel_rec.numpy() * 32767.0).astype('int16')

            # 构建输出音频路径
            output_path_i = os.path.join(
                out_path,
                f"wav_rec_{basename}.wav"
            )

            # 保存音频文件
            write(output_path_i, model.h.sampling_rate, gt_mel_rec_int16[0])  # 参数分别为路径、采样率、音频数据


# 输入和输出路径
file_path = '/sdc/data/zhaoyuan/V2C/Chenqi_Denoise/Phoneme_level_Feature/Denoise_version2_all_feature/val.txt'
gt_mel_path = '/sdc/data/zhaoyuan/V2C/Chenqi_Denoise/Phoneme_level_Feature/Denoise_version2_all_feature/mel'
out_path = '/sdc/data/zhaoyuan/V2C/Chenqi_Denoise/Phoneme_level_Feature/Denoise_version2_all_feature/setting1_gt_mel_rec_audio'

# 打印开始信息
print("Processing started...")
extract(file_path, gt_mel_path, out_path)
print("Processing completed!")
