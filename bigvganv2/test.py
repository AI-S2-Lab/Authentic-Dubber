import os
import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write

# 指定运行设备为 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化 BigVGAN 模型
model = bigvgan.BigVGAN.from_pretrained('/sdc/data/zhaoyuan/bigvgan_v2_22khz_80band_fmax8k_256x', use_cuda_kernel=False)

# 移除模型中的权重归一化，并设置为评估模式
model.remove_weight_norm()
model = model.eval().to(device)

# 加载 wav 文件并调整采样率为模型要求的采样率
wav_path = '/sdc/data/zhaoyuan/V2C/Chenqi_Denoise/Phoneme_level_Feature/Denoise_version2_all_feature/val_gt_audio_lab/Inside@Sadness_00_1119_00.wav'
wav, _ = librosa.load(wav_path, sr=22050, mono=True)  # wav 是一个 NumPy 数组，形状为 [T_time]，值域在 [-1, 1] 之间
wav = torch.FloatTensor(wav).unsqueeze(0)  # 将 wav 转为 FloatTensor，形状为 [B(1), T_time]

# 从原始音频中计算 mel 频谱图
mel = get_mel_spectrogram(wav, model.h).to(device)  # mel 是 FloatTensor，形状为 [B(1), C_mel, T_frame]

# 从 mel 频谱生成波形
with torch.inference_mode():  # 关闭梯度计算以加速推理
    wav_gen = model(mel)  # wav_gen 是 FloatTensor，形状为 [B(1), 1, T_time]，值域在 [-1, 1] 之间
wav_gen_float = wav_gen.squeeze(0).cpu()  # wav_gen 是 FloatTensor，形状为 [1, T_time]

# 将生成的波形音频转换为 16-bit 线性 PCM 格式
wav_gen_int16 = (wav_gen_float.numpy() * 32768.0).astype('int16')  # 转换为 NumPy 数组，形状为 [T_time]，数据类型为 int16

# 指定输出路径
output_dir = '/sdc/home/zhaoyuan/projects/bigvganv2/generated_audio'
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
output_path = os.path.join(output_dir, 'gen.wav')

# 使用 scipy 保存音频
write(output_path, 22050, wav_gen_int16[0])  # 参数分别为路径、采样率、音频数据

print(f"生成的音频已保存到 {output_path}")
