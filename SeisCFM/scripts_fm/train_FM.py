import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from configs.UNet import SeismicUNet  # 注意：UNet 的输出头已改为线性，无激活/Norm
from configs.train_parameter import CFG
from configs.tools import init_csv_logger, EarlyStopping, make_scheduler, EMA, noise_alpha, append_csv
from configs.Dataset import NGADataset
from scripts_fm.train_fm_one_epoch import train_fm_one_epoch
from scripts_fm.evaluate_fm import evaluate
from scripts_fm.evaluate_quality import evaluate_quality

def main():
    ckpt_path = os.path.join(CFG.SAVE_DIR, "last_s0.pt")

    stats_path = r"G:\GRX\GMA\Data\global_stats.npz"
    csv_path = r"G:\GRX\GMA\Data\meta.csv"
    h5_path = r"G:\GRX\GMA\Data\NGA_West2.hdf5"

    assert os.path.exists(csv_path) and os.path.exists(h5_path), "请检查数据路径"

    # === PATCH C1: CSV & EarlyStopping 初始化 ===
    metrics_csv = os.path.join(CFG.SAVE_DIR, "train_epoch_metrics.csv")
    init_csv_logger(metrics_csv, header=[
        "timestamp", "epoch", "train_loss", "val_loss", "cos", "|v|/|u|",
        "RMSE", "MAPE", "EnergyRatio", "BandL1", "KeyRatio", "noise_alpha"
    ])

    # patience 建议：max(10, int(0.2*N_EPOCHS))
    early_stop = EarlyStopping(patience=20, min_delta=1e-4, verbose=True)

    # 划分
    meta_df = pd.read_csv(csv_path, low_memory=False)
    eids = meta_df["filename"].unique()
    train_eid, temp_eid = train_test_split(eids, test_size=0.2, random_state=42)
    val_eid, test_eid = train_test_split(temp_eid, test_size=0.5, random_state=42)

    train_idx = meta_df.index[meta_df["filename"].isin(train_eid)].tolist()
    val_idx = meta_df.index[meta_df["filename"].isin(val_eid)].tolist()

    ds = NGADataset(h5_path=h5_path, csv_path=csv_path, stats_path=stats_path)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True,
                            pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicUNet(img_size=(129, 188), in_channels=3, out_channels=3).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": CFG.LEARNING_RATE, "weight_decay": CFG.WEIGHT_DECAY}
    ])
    total_steps = CFG.N_EPOCHS * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps)

    ema = EMA(model, decay=CFG.EMA_DECAY)

    fm = ConditionalFlowMatcher(sigma=CFG.SIGMA)
    # fm = ExactOptimalTransportConditionalFlowMatcher(sigma=CFG.SIGMA)

    writer = SummaryWriter(os.path.join(CFG.SAVE_DIR, "logs"))
    global_step = 0
    start_epoch = 1

    best_val = float("inf")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        ema.shadow = checkpoint["ema"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

        best_val = float(checkpoint.get("best_val", float("inf")))

        es = checkpoint["early_stop"]
        early_stop.best = float(es.get("best", best_val))
        early_stop.counter = int(es.get("counter", 0))
        early_stop.should_stop = bool(es.get("should_stop", False))

        if "cfg" in checkpoint:
            snap = checkpoint["cfg"]
            try:
                msg = []
                if snap.get("ema_decay", None) is not None and snap["ema_decay"] != CFG.EMA_DECAY:
                    msg.append(f"ema_decay: ckpt={snap['ema_decay']} now={CFG.EMA_DECAY}")
                if msg:
                    print("[Resume][Warn] Config mismatch:\n  - " + "\n  - ".join(msg))
            except Exception as e:
                print(f"[Resume][Warn] cfg check failed: {e}")

        print(f"[Resume] Loaded {ckpt_path}. Resume at epoch {start_epoch}, global_step {global_step}.")
    else:
        print("[Resume] No checkpoint found. Start from scratch.")

    for epoch in range(start_epoch, CFG.N_EPOCHS + 1):
        train_loss, train_cos, train_ratio, global_step = train_fm_one_epoch(model, train_loader, fm,
                                                  optimizer, scheduler, ema,
                                                  device, epoch, writer, global_step)

        val_loss = evaluate(model, val_loader, fm, device, epoch, writer)

        print(
            f"Epoch {epoch}/{CFG.N_EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | alpha: {noise_alpha(epoch):.2f}")

        qual = evaluate_quality(model, val_loader, device, writer, epoch,
                                save_dir=CFG.SAVE_DIR, n_batches=10)

        # —— 记录最优（可按 val_loss/val_cos/val_ratio 组合策略）——
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow,  # ← 保存 EMA
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": {
                    "img_size": (129, 188),
                    "in_ch": 3, "out_ch": 3,
                    "ema_decay": CFG.EMA_DECAY
                },
                "epoch": epoch,
                "global_step": global_step,
                "best_val": float(best_val),

                "early_stop": {
                    "best": float(early_stop.best),
                    "counter": int(early_stop.counter),
                    "should_stop": bool(getattr(early_stop, "should_stop", False)),
                },

            }, os.path.join(CFG.SAVE_DIR, "best_s0.pt"))

        # 始终保存最近一次
        torch.save({
            "model": model.state_dict(),
            "ema": ema.shadow,  # ← 保存 EMA
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": {
                "img_size": (129, 188),
                "in_ch": 3, "out_ch": 3,
                "ema_decay": CFG.EMA_DECAY
            },
            "epoch": epoch,
            "global_step": global_step,
            "best_val": float(best_val),

            "early_stop": {
                "best": float(early_stop.best),
                "counter": int(early_stop.counter),
                "should_stop": bool(getattr(early_stop, "should_stop", False)),
            },
        }, os.path.join(CFG.SAVE_DIR, "last_s0.pt"))

        rmse = qual["rmse"] if qual is not None else float("nan")
        mape = qual["mape"] if qual is not None else float("nan")
        enr = qual["energy_ratio"] if qual is not None else float("nan")
        bL1 = qual["band_L1"] if qual is not None else float("nan")
        krt = qual["key_ratio"] if qual is not None else float("nan")

        append_csv(metrics_csv, [
            datetime.now().isoformat(timespec="seconds"),
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_cos:.6f}", f"{train_ratio:.6f}",
            f"{rmse:.6f}", f"{mape:.6f}", f"{enr:.6f}", f"{bL1:.6f}", f"{krt:.6f}",
            f"{noise_alpha(epoch):.3f}"
        ])

        # === PATCH C3: 早停判定 ===
        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"[ES] Early stopping at epoch {epoch}. Best val={early_stop.best:.6f}")
            break

    writer.close()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
