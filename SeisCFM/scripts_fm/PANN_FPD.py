import torch
import seisbench.models as sbm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from configs.frechet_distance import frechet_from_embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from panns_inference import AudioTagging  # CNN14 PANNs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_sr = 32000
target_len = 32000 * 60
panns = AudioTagging(checkpoint_path=None, device="cuda")

wave_pre_dir = "G:\GRX\GMA\Data\wave_pre.npy"
wave_true_dir = "G:\GRX\GMA\Data\wave_true.npy"

wave_true = np.load(wave_true_dir)
wave_pre = np.load(wave_pre_dir)
# -----------------------------
#  工具函数：预处理 PANNs 输入
# -----------------------------
def _prepare_for_panns(waveforms, fs, dim):
    """
    waveforms: numpy 数组，形状 (N, C, T)，比如 (1872, 3, 6000)
    fs: 原始采样率，比如 100 Hz
    返回: (N, target_len)，单通道、重采样+补零/截断后的音频
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    N, C, T = waveforms.shape

    out = np.zeros((N, target_len), dtype=np.float32)

    for i in range(N):
        # (C, T) → (T,)，把三分量变成单通道
        x = waveforms[i][dim]

        # 重采样到 32 kHz
        if abs(fs - target_sr) > 1e-3:
            x = librosa.resample(x, orig_sr=fs, target_sr=target_sr)

        L = x.shape[0]

        # 截断 / 补零到固定长度
        if L >= target_len:
            x = x[:target_len]
        else:
            x = np.pad(x, (0, target_len - L), mode="constant")

        out[i] = x.astype(np.float32)

    return out

# -----------------------------
#  频域嵌入 (PANNs CNN14)
# -----------------------------
def extract_freq_embeddings(waveforms, fs, dim, batch_size=8):
    audio = _prepare_for_panns(waveforms, fs, dim)  # (N, L)
    N = audio.shape[0]

    all_embs = []

    for i in range(0, N, batch_size):
        batch = audio[i:i+batch_size]
        # panns_inference 要求形状 (batch, n_samples)
        # 它内部会做 log-mel 变换和池化
        clipwise_output, emb = panns.inference(batch)
        # emb: (batch, D_freq)
        all_embs.append(emb)

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

def eval_freq_FD(real_waveforms, fake_waveforms, fs, dim):
    """
    频域 FD: 在 PANNs 嵌入空间上的 Fréchet distance
    """
    emb_r = extract_freq_embeddings(real_waveforms, fs, dim)
    emb_f = extract_freq_embeddings(fake_waveforms, fs, dim)
    return frechet_from_embeddings(emb_r, emb_f)

freq_FD = eval_freq_FD(wave_true, wave_pre, 100, 2)

print("Freq-domain FD (PANNs CNN14):", freq_FD)

emb_real = extract_freq_embeddings(wave_true, 100, 2)
emb_fake = extract_freq_embeddings(wave_pre, 100, 2)

emb_all = np.concatenate([emb_real, emb_fake], axis=0)
labels = np.array([0] * len(emb_real) + [1] * len(emb_fake))  # 0=real,1=fake

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
emb_2d = tsne.fit_transform(emb_all)

# 把数值标签转成人类可读的
type_str = np.where(labels == 0, 'Real', 'Generated')

df = pd.DataFrame({
    'tsne_x': emb_2d[:, 0],
    'tsne_y': emb_2d[:, 1],
    'type': type_str,          # 或直接存 0/1 也行
    # 如果你有原始索引，也可以加一列 id
    # 'index': np.arange(len(emb_2d))
})

df.to_csv('tsne_pann_embeddings.csv', index=False, encoding='utf-8-sig')
print('保存完成：tsne_pann_embeddings.csv')

plt.figure()
plt.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], alpha=0.5, label='Real')
plt.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], alpha=0.5, label='Generated')
plt.legend()
plt.title('PANNs embedding space (t-SNE)')
plt.show()

