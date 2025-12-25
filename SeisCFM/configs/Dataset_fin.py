import numpy as np
import torch
from matplotlib.mlab import magnitude_spectrum
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from torch.utils.data import Dataset
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NGADataset(Dataset):
    def __init__(self, h5_path, csv_path, transforms=None, stats_path=r"G:\GRX\小论文地震动增强\Flow Matching\conditional-flow-matching-main\conditional-flow-matching-main\NGA_West2\global_stats.npz", gain=1.0):
        # 1. 读元数据表
        self.meta = pd.read_csv(csv_path, low_memory=False)

        # ——③ depth / magnitude / distance Z‑Score
        scaler = StandardScaler()
        self.meta[['Magnitude', 'Rrup', 'Vs30']] = scaler.fit_transform(
            self.meta[['Earthquake Magnitude', 'Rrup', 'Vs30 (m/s)']]
        )

        # 2. 构建索引列表
        self.trace_names = self.meta['filename'].tolist()

        # 3. 保存路径和可选 transforms
        self.h5_path = h5_path
        self.transforms = transforms

        # 4. HDF5 文件句柄留空，按需打开（避免多进程问题）&#8203;:contentReference[oaicite:4]{index=4}
        self._h5 = None

        stats = np.load(stats_path)
        self.GMU = float(stats["MU"])
        self.GSTD = float(stats["STD"])
        self.gain = gain

    def __len__(self):
        return len(self.trace_names)  # 返回样本数量 :contentReference[oaicite:5]{index=5}

    def __getitem__(self, idx):
        # 延迟打开 HDF5（每个 worker 会重新打开一次）
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        trace = self.meta.iloc[idx]['filename']
        spec = self._h5['spec_mag'][trace][()]
        wave = self._h5['waveform'][trace][()]

        spec_dB = 20.0 * np.log10(spec + 1e-8)

        spec_standard = (spec_dB - self.GMU) / self.GSTD  # 全局 z-score
        spec_standard *= self.gain  # 放大

        # 3. 转为 Tensor
        spec_tensor = torch.from_numpy(spec_dB).float()
        spec_standard_tensor = torch.from_numpy(spec_standard).float()

        # 4. 可选变换（归一化、数据增强等）
        if self.transforms:
            spec_tensor = self.transforms(spec_tensor)

        row = self.meta.iloc[idx]
        cond_tensor = torch.tensor([
            row['Magnitude'], row['Rrup'], row['Vs30']
        ], dtype=torch.float32)



        # 断层机制通常是一个离散标号 (例如 0~4)。如果从 CSV 中读取的是浮点值
        # (例如表示角度或编码)，这里将其转换为整数类型以便后续通过
        # ``nn.Embedding`` 查表。若值已经是整数，这里不会改变它的含义。
        fault_val = row['Mechanism Based on Rake Angle']
        # 将缺失或非数值值视为 0
        if pd.isnull(fault_val):
            fault_val = 0
        fault_type = torch.tensor([int(fault_val)], dtype=torch.long)

        return spec_tensor, cond_tensor, fault_type, wave

    # 关键：让 _h5 不参与 pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state
