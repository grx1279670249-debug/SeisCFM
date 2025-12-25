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

def main():
    model_dir = "Flow_Matching_2"
    SAVE_DIR = os.path.join("models", model_dir)
    ckpt_path = os.path.join(CFG.SAVE_DIR, "last_s0.pt")

    stats_path = r"G:\GRX\GMA\Data\global_stats.npz"
    csv_path = r"G:\GRX\GMA\Data\meta.csv"
    h5_path = r"G:\GRX\GMA\Data\NGA_West2.hdf5"
    assert os.path.exists(csv_path) and os.path.exists(h5_path), "请检查数据路径"

    stats = np.load(stats_path)
    GMU = float(stats["spec_mu_db"])
    GSTD = float(stats["spec_std_db"])

    # 划分
    meta_df = pd.read_csv(csv_path, low_memory=False)
    eids = meta_df["filename"].unique()
    train_eid, temp_eid = train_test_split(eids, test_size=0.2, random_state=42)
    val_eid, test_eid = train_test_split(temp_eid, test_size=0.5, random_state=42)

    train_idx = meta_df.index[meta_df["filename"].isin(train_eid)].tolist()
    test_idx = meta_df.index[meta_df["filename"].isin(test_eid)].tolist()

    ds = NGADataset(h5_path=h5_path, csv_path=csv_path, stats_path=stats_path)
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False,
                             pin_memory=True, drop_last=True)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False,
                            pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicUNet(img_size=(129, 188), in_channels=3, out_channels=3).to(device)
    total_steps = CFG.N_EPOCHS * len(train_loader)

    _ = load_fm_ckpt(model, os.path.join(SAVE_DIR, "best_s0.pt"), device, use_ema=True)

    stats = np.load(stats_path)
    mat_mean = float(stats.get("mat_mean"))
    mat_std = float(stats.get("mat_std"))
    rup_mean = float(stats.get("rup_mean"))
    rup_std = float(stats.get("rup_std"))
    vs_mean = float(stats.get("vs_mean"))
    vs_std = float(stats.get("vs_std"))

    for spec, meta, fault, wave, trace in train_loader:
        print(trace)
        spec = spec.to(device)
        meta = meta.to(device)
        fault = fault.to(device)
        z = torch.randn_like(spec)
        print("震级:",meta[0, 0]*mat_std + mat_mean)
        print("Rrup:", meta[0, 1]*rup_std + rup_mean)
        print("vs30:", meta[0, 2]*vs_std + vs_mean)
        print("fault:", fault)

        x_gen = short_ode_sample(model, meta, fault, device, steps=100, x_T_shape=spec.shape[1:])
        x_gen_2 = short_ode_sample(model, meta, fault, device, steps=100, x_T_shape=spec.shape[1:])
        x_gen_3 = short_ode_sample(model, meta, fault, device, steps=100, x_T_shape=spec.shape[1:])

        spec_db = spec * GSTD + GMU  # 反标准化回 dB 幅度谱
        mag = torch.pow(10.0, spec_db / 20.0) - 1e-8  # dB -> 线性幅度（下限用 clamp）
        wave_true = griffin_lim_reconstruct(
            mag,  # (n, C, F, T) 线性幅度
            n_fft=256,
            hop_length=32,
            win_length=256,  # 若数据集当初就是 256 窗，务必显式给出
            n_iter=100,
            # target_len=6000,  # 若你的函数支持该参数，**一定**要传
            device=device
        )

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

        plot_input_output_batch(spec_db, spec_pre_db)

        # plot_waveforms(wave_pre, max_samples=1)  # 自己给个开关

        plot_waveforms(wave_true, max_samples=1)
        plot_waveforms(wave_pre, max_samples=1)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()







