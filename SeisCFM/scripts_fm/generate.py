import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from configs.train_parameter import CFG
from configs.Dataset import NGADataset
from configs.UNet import SeismicUNet
from configs.tools import load_fm_ckpt, short_ode_sample, griffin_lim_reconstruct
from configs.visual_tools import plot_input_output_batch, plot_waveforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stats_path = r"G:\GRX\GMA\Data\global_stats.npz"
csv_path = r"G:\GRX\GMA\Data\meta.csv"
h5_path = r"G:\GRX\GMA\Data\NGA_West2.hdf5"

stats = np.load(stats_path)
mat_mean = float(stats.get("mat_mean"))
mat_std = float(stats.get("mat_std"))
rup_mean = float(stats.get("rup_mean"))
rup_std = float(stats.get("rup_std"))
vs_mean = float(stats.get("vs_mean"))
vs_std = float(stats.get("vs_std"))

def build_condition_batch(mw_arr, rrup_arr, vs30_arr, fault_arr):
    """
    输入：numpy 数组或 list，长度为 N
        mw_arr:       (N,)
        rrup_arr:     (N,)
        vs30_arr:     (N,)
        fault_arr:    (N,)  例如 0,1,2,3 对应不同断层类型
    输出：
        meta_batch:   (N, 3)  [Mw, Rrup, Vs30] 归一化后
        fault_batch:  (N, 1)
    """

    mw_arr = np.asarray(mw_arr, dtype=np.float32)
    rrup_arr = np.asarray(rrup_arr, dtype=np.float32)
    vs30_arr = np.asarray(vs30_arr, dtype=np.float32)
    fault_arr = np.asarray(fault_arr, dtype=np.float32)

    # ---------- 1. 按你训练时的统计量做归一化 ----------
    Mw = (mw_arr - mat_mean) / mat_std
    Rrup = (rrup_arr - rup_mean) / rup_std
    Vs30 = (vs30_arr - vs_mean) / vs_std

    meta = np.stack([Mw, Rrup, Vs30], axis=1)    # (N, 3)

    # ---------- 2. 转成 torch 张量并搬到 device ----------
    meta_batch = torch.from_numpy(meta).to(device)               # (N, 3)
    fault_batch = torch.from_numpy(fault_arr).reshape(-1, 1).to(device)  # (N, 1)

    return meta_batch, fault_batch


def main():
    model_dir = "Flow_Matching_2"
    SAVE_DIR = os.path.join("models", model_dir)
    OUT_DIR = "G:\GRX\GMA\Data\select_wave_pre_3.npy"

    GMU = float(stats["spec_mu_db"])
    GSTD = float(stats["spec_std_db"])

    # 你训练时 UNet 的输入尺寸
    IMG_SIZE = (129, 188)  # (F, T)
    IN_CH = 3
    OUT_CH = 3

    meta = pd.read_csv(csv_path, low_memory=False)
    mask = (
            meta['Earthquake Magnitude'].between(6, 7.5) &  # Mw ∈ [5, 6]
            meta['Rrup'].between(5, 50) &  # Rrup ∈ [10, 50]
            meta['Vs30 (m/s)'].between(360, 800)  # Vs30 ∈ [100, 250]
    )
    mw = meta.loc[mask, 'Earthquake Magnitude'].tolist()
    rrup = meta.loc[mask, 'Rrup'].tolist()
    vs30 = meta.loc[mask, 'Vs30 (m/s)'].tolist()
    fault_type = meta.loc[mask, 'Mechanism Based on Rake Angle'].tolist()

    meta_cond, fault_cond = build_condition_batch(mw, rrup, vs30, fault_type)
    model = SeismicUNet(
        img_size=IMG_SIZE,
        in_channels=IN_CH,
        out_channels=OUT_CH
    ).to(device)

    # 这里用 EMA 权重加载与你测试脚本一致
    _ = load_fm_ckpt(model, os.path.join(SAVE_DIR, "best_s0.pt"), device, use_ema=True)

    # # x_T_shape 需要的是 (C, F, T)，和训练时 spec.shape[1:] 一致
    x_T_shape = (IN_CH, IMG_SIZE[0], IMG_SIZE[1])

    batch_size = 16
    n_total = meta_cond.size(0)
    print(f"Total samples to generate: {n_total}")

    waves_list = []  # 用来存所有 batch 的结果

    model.eval()
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            meta_batch = meta_cond[start:end]  # (B, 3)
            fault_batch = fault_cond[start:end]  # (B, 1)

            print(f"Generating batch [{start}:{end}) ...")

            x_gen = short_ode_sample(model, meta_batch, fault_batch, device, steps=100, x_T_shape=x_T_shape)

            spec_pre_db = x_gen * GSTD + GMU  # 反标准化回 dB 幅度谱
            mag_pre = torch.pow(10.0, spec_pre_db / 20.0) - 1e-8
            wave_pre = griffin_lim_reconstruct(
                mag_pre,  # (n, C, F, T) 线性幅度
                n_fft=256,
                hop_length=32,
                win_length=256,  # 若数据集当初就是 256 窗，务必显式给出
                n_iter=100,
                # target_len=6000,  # 若你的函数支持该参数，**一定**要传
                device=device
            )

            waves_list.append(wave_pre.cpu())

    # ================== 5. 拼接全部 batch，并保存 ==================
    waves_all = torch.cat(waves_list, dim=0)  # (N, C, T) / (N, T)
    print(waves_all.size())
    wave_np = waves_all.numpy()
    # wave_np = wave_pre.cpu().numpy()
    np.save(OUT_DIR, wave_np)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

