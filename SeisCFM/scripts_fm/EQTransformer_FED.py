import torch
import seisbench.models as sbm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from configs.frechet_distance import frechet_from_embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_model = sbm.EQTransformer.from_pretrained("stead").to(device)
eqt_fs = time_model.sampling_rate
eqt_len = time_model.in_samples

wave_pre_dir = "G:\GRX\GMA\Data\wave_pre.npy"
wave_true_dir = "G:\GRX\GMA\Data\wave_true.npy"

wave_true = np.load(wave_true_dir)
wave_pre = np.load(wave_pre_dir)
# wave_pre = np.random.rand(1872, 3, 6000)
# N = wave_pre.shape[0]
# C = np.concatenate([wave_pre, wave_true], axis=0)
# perm = np.random.permutation(2*N)
# C_shuffled = C[perm]
# wave_true = C_shuffled[:N]
# wave_pre = C_shuffled[N:]


def _prepare_for_time_model(waveforms):
    out = np.stack([waveforms[:, 2, :], waveforms[:, 0, :], waveforms[:, 1, :]], axis=1)
    tensor = torch.from_numpy(out).float()
    return tensor

def extract_time_embeddings(waveforms, batch_size=32):

    time_model.eval()
    tensor = _prepare_for_time_model(waveforms)  # (N, 3, L)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embs = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)

            # 用 SeisBench 自带的归一化逻辑 (annotate_batch_pre)，保证和训练时一致
            batch_norm = time_model.annotate_batch_pre(batch, argdict={})

            # 手动走到 "transformer_d" 之后，拿特征 x
            x = time_model.encoder(batch_norm)
            x = time_model.res_cnn_stack(x)
            x = time_model.bi_lstm_stack(x)
            x, _ = time_model.transformer_d0(x)
            x, _ = time_model.transformer_d(x)
            # x 形状: (batch, C, T_down)

            # 简单做全局平均池化，得到固定维度嵌入
            emb = torch.mean(x, dim=2)  # (batch, C)
            all_embs.append(emb.cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

#  外部接口：直接算 FD
# -----------------------------
def eval_time_FID(real_waveforms, fake_waveforms):
    """
    时域 FID: 在 EQTransformer/PhaseNet 嵌入空间上的 Fréchet distance
    """
    emb_r = extract_time_embeddings(real_waveforms)
    emb_f = extract_time_embeddings(fake_waveforms)
    return frechet_from_embeddings(emb_r, emb_f)

time_FID = eval_time_FID(wave_true, wave_pre)

print("Time-domain FID (EQTransformer):", time_FID)

emb_real = extract_time_embeddings(wave_true)
emb_fake = extract_time_embeddings(wave_pre)
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

df.to_csv('tsne_eqt_embeddings.csv', index=False, encoding='utf-8-sig')
print('保存完成：tsne_eqt_embeddings.csv')

plt.figure()
plt.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], alpha=0.5, label='Real')
plt.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], alpha=0.5, label='Generated')
plt.legend()
plt.title('EQTransformer embedding space (t-SNE)')
plt.show()