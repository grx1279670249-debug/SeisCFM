import numpy as np
import torch
from matplotlib.mlab import magnitude_spectrum
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from torch.utils.data import Dataset
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NGADataset(Dataset):
    def __init__(self, h5_path, csv_path, stats_path):
        # 1. 读元数据表
        self.meta = pd.read_csv(csv_path, low_memory=False)

        # 2. 构建索引列表
        self.trace_names = self.meta['filename'].tolist()

        # 3. 保存路径和可选 transforms
        self.h5_path = h5_path

        # 4. HDF5 文件句柄留空，按需打开（避免多进程问题）&#8203;:contentReference[oaicite:4]{index=4}
        self._h5 = None

        stats = np.load(stats_path)

        # 兼容你的原始键名：若没有 *_db 键，就回落到 MU/STD
        self.spec_mu_db = float(stats.get("spec_mu_db"))
        self.spec_std_db = float(stats.get("spec_std_db"))
        # 保护下限，避免除 0
        self.spec_std_db = float(max(self.spec_std_db, 1e-8))

        self.mat_mean = float(stats.get("mat_mean"))
        self.mat_std = float(stats.get("mat_std"))

        self.rup_mean = float(stats.get("rup_mean"))
        self.rup_std = float(stats.get("rup_std"))

        self.vs_mean = float(stats.get("vs_mean"))
        self.vs_std = float(stats.get("vs_std"))

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        # 延迟打开 HDF5（每个 worker 会重新打开一次）
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        trace = self.meta.iloc[idx]['filename']
        spec = self._h5['spec_mag'][trace][()]
        wave = self._h5['waveform'][trace][()]

        spec_db = 20.0 * np.log10(spec + 1e-8)
        # ——谱标准化（训练集常数；默认供模型使用）
        spec_std = (spec_db - self.spec_mu_db) / self.spec_std_db

        # 3. 转为 Tensor
        spec_tensor = torch.from_numpy(spec_std).float()

        row = self.meta.iloc[idx]
        cond_mw = torch.tensor(row['Earthquake Magnitude'], dtype=torch.float32)
        mw_std = (cond_mw - self.mat_mean) / self.mat_std

        cond_rrup = torch.tensor(row['Rrup'], dtype=torch.float32)
        rrup_std = (cond_rrup - self.rup_mean) / self.rup_std

        cond_vs30 = torch.tensor(row['Vs30 (m/s)'], dtype=torch.float32)
        vs30_std = (cond_vs30 - self.vs_mean) / self.vs_std

        cond_tensor = torch.tensor([mw_std, rrup_std, vs30_std], dtype=torch.float32)

        fault_val = row['Mechanism Based on Rake Angle']
        if pd.isnull(fault_val):
            fault_val = 0
        fault_type = torch.tensor([int(fault_val)], dtype=torch.long)

        return spec_tensor, cond_tensor, fault_type, wave, trace

    # 关键：让 _h5 不参与 pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state