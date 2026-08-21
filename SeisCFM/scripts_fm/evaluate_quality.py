import torch
from pathlib import Path
from configs.tools import short_ode_sample
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
def proxy_metrics(
        gen_spec_std: torch.Tensor,
        real_spec_std: torch.Tensor,
        spec_mu_db: float,
        spec_std_db: float,
        magnitude_floor: float = 1e-8,
):
    """Compare normalized spectra without treating normalized dB values as energy."""
    assert gen_spec_std.shape == real_spec_std.shape, "gen/real spec shape mismatch"
    if spec_std_db <= 0:
        raise ValueError("spec_std_db must be positive")

    diff_std = gen_spec_std - real_spec_std
    rmse_std = torch.sqrt((diff_std ** 2).mean(dim=(1, 2, 3)))

    gen_db = gen_spec_std * spec_std_db + spec_mu_db
    real_db = real_spec_std * spec_std_db + spec_mu_db
    mae_db = (gen_db - real_db).abs().mean(dim=(1, 2, 3))

    # Dataset.py uses 20*log10(magnitude + 1e-8); invert that transform
    # before computing physical energy and band-share metrics.
    gen_mag = (torch.pow(10.0, gen_db / 20.0) - magnitude_floor).clamp_min(0.0)
    real_mag = (torch.pow(10.0, real_db / 20.0) - magnitude_floor).clamp_min(0.0)
    tiny = torch.finfo(gen_mag.dtype).tiny

    gE = (gen_mag ** 2).sum(dim=(1, 2, 3))
    rE = (real_mag ** 2).sum(dim=(1, 2, 3))
    energy_ratio = gE / rE.clamp_min(tiny)

    # 频带能量占比
    freqs = np.linspace(0, 50, gen_mag.size(2))
    gL, gM, gH = split_bands(gen_mag, freqs)  # (B,C,*,W)
    rL, rM, rH = split_bands(real_mag, freqs)

    def band_share(xL, xM, xH):
        EL = (xL**2).sum(dim=(1,2,3))
        EM = (xM**2).sum(dim=(1,2,3))
        EH = (xH**2).sum(dim=(1,2,3))
        S = (EL + EM + EH).clamp_min(tiny)
        return EL/S, EM/S, EH/S  # (B,)

    gSL, gSM, gSH = band_share(gL, gM, gH)
    rSL, rSM, rSH = band_share(rL, rM, rH)

    # 三段带能占比差（L1）
    band_L1 = (gSL-rSL).abs() + (gSM-rSM).abs() + (gSH-rSH).abs()  # (B,)

    # 关键频带局部能量（可根据你的T1映射到频率带后替换索引范围）
    # 这里以 1-10 Hz 中频段作为桥梁敏感代理；也可细分一个更窄的窗口
    g_key = (gM ** 2).sum(dim=(1,2,3))
    r_key = (rM ** 2).sum(dim=(1,2,3))
    key_ratio = g_key / r_key.clamp_min(tiny)

    # 汇总为标量（batch均值）
    return {
        "rmse_std": rmse_std.mean().item(),
        "mae_db": mae_db.mean().item(),
        "energy_ratio": energy_ratio.mean().item(),
        "band_L1": band_L1.mean().item(),
        "key_ratio": key_ratio.mean().item()
    }

@torch.no_grad()
def evaluate_quality(
        model, loader, device, writer, epoch, save_dir, n_batches,
        spec_mu_db, spec_std_db,
):

    model.eval()
    metrics_accum = {
        "rmse_std": 0.0,
        "mae_db": 0.0,
        "energy_ratio": 0.0,
        "band_L1": 0.0,
        "key_ratio": 0.0,
    }
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
        m = proxy_metrics(gen, real, spec_mu_db, spec_std_db)
        for k in metrics_accum:
            metrics_accum[k] += m[k]
        count += 1

    if count > 0:
        for k in metrics_accum: metrics_accum[k] /= count
        if writer is not None:
            writer.add_scalar("qual/rmse_std", metrics_accum["rmse_std"], epoch)
            writer.add_scalar("qual/mae_db", metrics_accum["mae_db"], epoch)
            writer.add_scalar("qual/energy_ratio", metrics_accum["energy_ratio"], epoch)
            writer.add_scalar("qual/band_L1", metrics_accum["band_L1"], epoch)
            writer.add_scalar("qual/key_ratio", metrics_accum["key_ratio"], epoch)

        print(f"[Qual] ep{epoch}: RMSE(std)={metrics_accum['rmse_std']:.4f} | "
              f"MAE(dB)={metrics_accum['mae_db']:.4f} | EnergyRatio={metrics_accum['energy_ratio']:.3f} | "
              f"BandL1={metrics_accum['band_L1']:.3f} | KeyRatio={metrics_accum['key_ratio']:.3f}")

        return metrics_accum
    else:
        return None
