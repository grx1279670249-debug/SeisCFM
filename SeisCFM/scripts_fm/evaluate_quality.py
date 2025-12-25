import torch
from pathlib import Path
from configs.tools import short_ode_sample
import pandas as pd
import numpy as np

# ---------- 频带与代理指标 ----------
def split_bands(t: torch.Tensor, freqs: np.ndarray, freq_dim=2):

    assert len(freqs) == t.size(freq_dim), "freqs长度必须与频率维一致"

    # 找到各频段索引边界
    b1 = np.searchsorted(freqs, 1.0)
    b2 = np.searchsorted(freqs, 10.0)
    b0 = np.searchsorted(freqs, 0.1)

    sl_all = [slice(None)] * t.dim()
    sl_low = sl_all.copy()
    sl_low[freq_dim] = slice(b0, b1)
    sl_mid = sl_all.copy()
    sl_mid[freq_dim] = slice(b1, b2)
    sl_high = sl_all.copy()
    sl_high[freq_dim] = slice(b2, None)

    return t[tuple(sl_low)], t[tuple(sl_mid)], t[tuple(sl_high)]


@torch.no_grad()
def proxy_metrics(gen_spec: torch.Tensor, real_spec: torch.Tensor, eps=1e-8):
    # 对齐形状
    assert gen_spec.shape == real_spec.shape, "gen/real spec shape mismatch"
    B = gen_spec.size(0)

    # 逐像素 RMSE / MAPE（对幅值谱/对数谱均可；你的数据目前是标准化后的spec_std，保持一致比较）
    diff = gen_spec - real_spec
    rmse = torch.sqrt((diff ** 2).mean(dim=(1,2,3)))          # (B,)
    mape = (diff.abs() / (real_spec.abs() + eps)).mean(dim=(1,2,3))

    # 能量（平方和）
    gE = (gen_spec ** 2).sum(dim=(1,2,3))
    rE = (real_spec ** 2).sum(dim=(1,2,3))
    energy_ratio = gE / (rE + eps)

    # 频带能量占比
    freqs = np.linspace(0, 50, 129)
    gL, gM, gH = split_bands(gen_spec, freqs)  # (B,C,*,W)
    rL, rM, rH = split_bands(real_spec, freqs)

    def band_share(xL, xM, xH, eps=1e-8):
        EL = (xL**2).sum(dim=(1,2,3))
        EM = (xM**2).sum(dim=(1,2,3))
        EH = (xH**2).sum(dim=(1,2,3))
        S  = EL + EM + EH + eps
        return EL/S, EM/S, EH/S  # (B,)

    gSL, gSM, gSH = band_share(gL, gM, gH)
    rSL, rSM, rSH = band_share(rL, rM, rH)

    # 三段带能占比差（L1）
    band_L1 = (gSL-rSL).abs() + (gSM-rSM).abs() + (gSH-rSH).abs()  # (B,)

    # 关键频带局部能量（可根据你的T1映射到频率带后替换索引范围）
    # 这里以“低频段”作为桥梁敏感代理；也可细分一个更窄的窗口
    g_key = (gM ** 2).sum(dim=(1,2,3))
    r_key = (rM ** 2).sum(dim=(1,2,3))
    key_ratio = g_key / (r_key + eps)

    # 汇总为标量（batch均值）
    return {
        "rmse": rmse.mean().item(),
        "mape": mape.mean().item(),
        "energy_ratio": energy_ratio.mean().item(),
        "band_L1": band_L1.mean().item(),
        "key_ratio": key_ratio.mean().item()
    }

@torch.no_grad()
def evaluate_quality(model, loader, device, writer, epoch, save_dir, n_batches):

    model.eval()
    metrics_accum = {"rmse": 0.0, "mape": 0.0, "energy_ratio": 0.0, "band_L1": 0.0, "key_ratio": 0.0}
    count = 0

    if save_dir is not None:
        save_root = Path(save_dir) / f"samples_ep{epoch:03d}"
        save_root.mkdir(parents=True, exist_ok=True)

    it = 0
    for spec, meta, fault, wave, _ in loader:
        it += 1
        if it > n_batches:
            break

        meta = meta.to(device)
        fault = fault.to(device)

        gen = short_ode_sample(model, meta, fault, device, steps=20, x_T_shape=spec.shape[1:])

        real = spec.to(device)
        m = proxy_metrics(gen, real)
        for k in metrics_accum:
            metrics_accum[k] += m[k]
        count += 1

    if count > 0:
        for k in metrics_accum: metrics_accum[k] /= count
        if writer is not None:
            writer.add_scalar("qual/rmse", metrics_accum["rmse"], epoch)
            writer.add_scalar("qual/mape", metrics_accum["mape"], epoch)
            writer.add_scalar("qual/energy_ratio", metrics_accum["energy_ratio"], epoch)
            writer.add_scalar("qual/band_L1", metrics_accum["band_L1"], epoch)
            writer.add_scalar("qual/key_ratio", metrics_accum["key_ratio"], epoch)

        print(f"[Qual] ep{epoch}: RMSE={metrics_accum['rmse']:.4f} | "
              f"MAPE={metrics_accum['mape']:.4f} | EnergyRatio={metrics_accum['energy_ratio']:.3f} | "
              f"BandL1={metrics_accum['band_L1']:.3f} | KeyRatio={metrics_accum['key_ratio']:.3f}")

        return metrics_accum
    else:
        return None