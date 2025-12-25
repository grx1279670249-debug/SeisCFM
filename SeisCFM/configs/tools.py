import numpy as np

from configs.train_parameter import CFG
import torch
import os
import csv
from math import cos, pi
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchdiffeq
from scipy.signal import stft
import scipy.signal as sg

# 噪声从0→1的过渡轮数
def noise_alpha(epoch):
    if epoch <= 0:
        return 0
    if epoch >= CFG.CURRICULUM_EPOCHS:
        return 1.0
    return epoch / float(CFG.CURRICULUM_EPOCHS)

def cosine_similarity_batch(u: torch.Tensor, v: torch.Tensor, eps=1e-8):
    u_flat = u.view(u.size(0), -1)
    v_flat = v.view(v.size(0), -1)
    num = torch.sum(u_flat * v_flat, dim=1)
    den = torch.linalg.norm(u_flat, dim=1) * torch.linalg.norm(v_flat, dim=1) + eps
    return num / den  # (B,)

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

def init_csv_logger(csv_path: str, header: list[str]):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if is_new:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float):
        if value + self.min_delta < self.best:
            self.best = value
            self.counter = 0
            if self.verbose:
                print(f"[ES] New best val={value:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[ES] No improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True

# def make_scheduler(optim: torch.optim.Optimizer, total_steps: int, min_factor=1e-2):
#
#     warmup_steps = CFG.WARMUP_STEPS
#
#     def lr_lambda(step: int):
#         if step < warmup_steps:
#             step = min(step, total_steps)
#             return float(step) / float(max(1, warmup_steps))
#         prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         return min_factor + (1-min_factor) * 0.5 * (1.0 + cos(pi * prog))
#     return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

def make_scheduler(optim: torch.optim.Optimizer, total_steps: int):
    def lr_lambda(step: int):
        if step < CFG.WARMUP_STEPS:
            return float(step) / float(max(1, CFG.WARMUP_STEPS))
        prog = (step - CFG.WARMUP_STEPS) / float(max(1, total_steps - CFG.WARMUP_STEPS))
        return 0.5 * (1.0 + cos(pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

def append_csv(csv_path: str, row: list):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def stft_torch(
        wave,
        n_fft,
        win_length,
        device
):
    # -------- 设备与窗函数 --------
    if device is None:
        device = wave.device

    wave_batch = wave.to(device)
    if win_length is None:
        win_length = n_fft
    window = torch.hann_window(win_length, device=device)

    B, C, time = wave_batch.shape  # (B, 3, T)
    wave_flat = wave_batch.reshape(B * C, time)

    # -------- Griffin-Lim 主体：从幅度谱恢复时域 --------
    spec_flat = torch.stft(
        wave_flat,
        n_fft=256,
        hop_length=32,
        win_length=256,
        window=window,
        return_complex=True
    )
    F, Frames = spec_flat.shape[-2], spec_flat.shape[-1]
    spec = spec_flat.view(B, C, F, Frames)

    return spec

def griffin_lim_reconstruct(
        mag_batch: torch.Tensor,
        n_fft: int = 256,
        hop_length: int = 32,
        win_length: Union[int, None] = None,
        n_iter: int = 32,
        momentum: float = 0.99,
        target_len: Union[int, None] = None,
        device: Union[str, torch.device, None] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    # -------- 设备与窗函数 --------
    if device is None:
        device = mag_batch.device

    mag_batch = mag_batch.to(device)
    if win_length is None:
        win_length = n_fft
    window = torch.hann_window(win_length, device=device)

    # -------- 形状展开 --------
    B, C, Freq, Frames = mag_batch.shape  # (B, 3, F, T)
    assert C == 3, "预期三分量 (C=3)"
    assert Freq == n_fft // 2 + 1, f"Freq={Freq} 与 n_fft 不匹配，应为 {n_fft//2+1}"

    mag_flat = mag_batch.reshape(B * C, Freq, Frames)

    # -------- Griffin-Lim 主体：从幅度谱恢复时域 --------
    waves_flat = AF.griffinlim(
        specgram=mag_flat,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        power=1.0,          # 线性幅度
        n_iter=n_iter,
        momentum=momentum,
        length=target_len,  # 直接控制输出长度
        rand_init=False
    )
    waves = waves_flat.view(B, C, -1).cpu()  # (B, 3, time) —— 与原实现一致

    return waves

def load_fm_ckpt(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    use_ema: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):

    ckpt = torch.load(path, map_location=device)
    report = {"path": path}

    # ---- 主干：先加载基础权重 ----
    missing_m, unexpected_m = model.load_state_dict(ckpt.get("model", {}), strict=False)
    report["model_missing"] = missing_m
    report["model_unexpected"] = unexpected_m

    # ---- 可选：覆写 EMA 到主干 ----
    if use_ema and ("ema" in ckpt) and ckpt["ema"]:
        # ckpt["ema"] 是 shadow dict（仅浮点参数），与 model.state_dict() 键空间一致/子集
        state = model.state_dict()
        # 只更新存在于 state 的键，避免无关键报错
        for k, v in ckpt["ema"].items():
            if k in state and torch.is_tensor(state[k]) and torch.is_tensor(v):
                state[k] = v.to(device)
        model.load_state_dict(state, strict=False)
        report["ema_applied"] = True
    else:
        report["ema_applied"] = False

    # ---- 优化器/调度器（可选，若断点续训）----
    if ("optimizer" in ckpt) and (optimizer is not None):
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            report["optimizer_loaded"] = True
        except Exception as e:
            report["optimizer_loaded"] = f"failed: {e}"
    else:
        report["optimizer_loaded"] = False

    if ("scheduler" in ckpt) and (scheduler is not None):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            report["scheduler_loaded"] = True
        except Exception as e:
            report["scheduler_loaded"] = f"failed: {e}"
    else:
        report["scheduler_loaded"] = False

    # 将模型转到目标 device（即使 map_location 已做，也无伤大雅）
    model.to(device)

    return report

@torch.no_grad()
def short_ode_sample(model, meta, fault, device, steps=7, x_T_shape=(3, 129, 188)):
    B = meta.size(0)
    x_T = torch.randn(B, *x_T_shape, device=device)
    def vf(t, x):
        t_exp = torch.full((B,), t, device=device)
        return model(t_exp, x, meta, fault)
    t_span = torch.linspace(0.0, 1.0, steps, device=device)
    traj = torchdiffeq.odeint(vf, x_T, t_span, atol=1e-4, rtol=1e-4, method="dopri5")
    # traj = torchdiffeq.odeint(vf, x_T, t_span, method="rk4",options={"step_size": float(t_span[1]-t_span[0])})
    return traj[-1]   # x_gen


def calculate_pga(waves_dir):
    waves = np.load(waves_dir)
    pga = np.nanmax(np.abs(waves), axis=2, keepdims=True)  # (N,3,1)
    return pga


def _newmark_sdof_response(acc_g, dt, wn, zeta=0.05):
    """
    Newmark-β(γ=1/2, β=1/4) 计算相对位移/速度/加速度响应
    """
    L = acc_g.shape[0]
    u = np.zeros(L)
    v = np.zeros(L)
    a = np.zeros(L)

    m = 1.0
    k = wn**2 * m
    c = 2.0 * zeta * wn * m

    gamma = 0.5
    beta = 0.25

    a0 = m/(beta*dt*dt) + gamma*c/(beta*dt)
    a1 = m/(beta*dt) + c*(gamma/beta - 1.0)
    a2 = m*(1.0/(2*beta) - 1.0) + c*dt*(gamma/(2*beta) - 1.0)

    k_eff = k + a0

    # 等效外力 p = -m*acc_g (m=1)
    p = -acc_g.astype(float)
    a[0] = (p[0] - c * v[0] - k * u[0]) / m

    for i in range(L-1):
        p_eff = p[i+1] + a0*u[i] + a1*v[i] + a2*a[i]

        # 这里实际上是 u_{i+1}，变量名叫 du 有点误导，但数学上没问题
        u[i+1] = p_eff / k_eff

        v[i+1] = (gamma/(beta*dt))*(u[i+1]-u[i]) \
                 + (1 - gamma/beta)*v[i] \
                 + dt*(1 - gamma/(2*beta))*a[i]

        # ★ 关键修正在这里
        a[i+1] = (1.0/(beta*dt*dt))*(u[i+1]-u[i]) \
                 - (1.0/(beta*dt))*v[i] \
                 - (1.0/(2*beta) - 1.0)*a[i]

    return u, v, a


def calculate_sa_T(waves_dir, dt, T=1.0, zeta=0.05, mode="psa", return_geom_mean=True):
    """
        从时域加速度计算 Sa(T, zeta)，默认输出 PSA。
        waves_dir: npy 文件路径，内含 (N,3,L) 的地震动加速度
        dt: 采样间隔(s)
        T:  目标周期(s)，默认为 1.0s
        zeta: 阻尼比，默认 5% (=0.05)
        mode: "psa" 或 "asa"；psa=ωn^2*max|u|；asa=max|u_ddot + acc_g|
        return_geom_mean: 返回两水平分量几何均值（形状 (N,1)）
        返回:
          sa: (N,3,1) 对应三个分量的谱加速度
          sa_gm: (N,1) 两水平(0/1)分量几何均值（若 return_geom_mean=True）
        """
    waves = np.load(waves_dir)  # (N,3,L)
    assert waves.ndim == 3 and waves.shape[1] == 3, "waves 应为 (N,3,L)"
    N, C, L = waves.shape

    wn = 2.0 * np.pi / T

    sa = np.zeros((N, C, 1), dtype=np.float64)

    for n in range(N):
        for c in range(C):
            acc_g = waves[n, c, :].astype(np.float64)

            # 相对响应
            u, v, a_rel = _newmark_sdof_response(acc_g, dt, wn, zeta)

            # 伪谱加速度 PSA = ωn^2 * max|u|
            sa_val = (wn ** 2) * np.nanmax(np.abs(u))

            sa[n, c, 0] = sa_val

    return sa

def calculate_sa_1s(waves_dir, dt, zeta=0.05, mode="psa"):
    sa = calculate_sa_T(waves_dir, dt, T=1.0, zeta=zeta, mode=mode, return_geom_mean=False)
    return sa  # (N,3,1)

def calculate_sa(waves_dir, T, dt, zeta=0.05, mode="psa"):
    sa = calculate_sa_T(waves_dir, dt, T=T, zeta=zeta, mode=mode, return_geom_mean=False)
    return sa  # (N,3,1)




