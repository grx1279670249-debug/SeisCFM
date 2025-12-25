import matplotlib.pyplot as plt
import numpy as np
import os

def plot_input_output_batch(input_batch, output_batch, index=0):
    """
    输入和输出谱图对比
    input_batch, output_batch: torch.Tensor, 形状 (B, 3, 129, 189)
    index: 要显示的 batch 索引
    """
    # 取出指定样本
    input_spec = input_batch[index].detach().cpu().numpy()
    output_spec = output_batch[index].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(3):
        # 输入谱
        im0 = axes[0, i].imshow(
            input_spec[i],
            origin="lower", aspect="auto", cmap="viridis"
        )
        axes[0, i].set_title(f"Input - Component {i + 1}")
        fig.colorbar(im0, ax=axes[0, i])

        # 输出谱
        im1 = axes[1, i].imshow(
            output_spec[i],
            origin="lower", aspect="auto", cmap="viridis"
        )
        axes[1, i].set_title(f"Output - Component {i + 1}")
        fig.colorbar(im1, ax=axes[1, i])

    plt.tight_layout()
    plt.show()

def plot_waveforms(
        waves,                 # (B, 3, 6000)  torch.Tensor 或 np.ndarray
        sr: int = 100,         # 采样频率 (Hz)
        max_samples: int = 5,  # 最多展示 / 保存多少条样本
        save_dir=None          # 若给路径，则保存 PNG；否则直接 plt.show()
):
    """
    绘制三通道地震波形，每个通道独立坐标系。
    - waves       : (B, 3, N) 张量或数组
    - sr          : 采样率
    - max_samples : 限制显示/保存的样本数S
    - save_dir    : None ⇒ 显示；否则按 sample_i.png 保存
    """
    # 转成 NumPy
    waves_np = (waves.detach().cpu().numpy()
                if hasattr(waves, "detach") else np.asarray(waves))
    B, C, N = waves_np.shape
    t = np.arange(N) / sr      # 时间轴 (秒)

    for i in range(min(B, max_samples)):
        fig, axes = plt.subplots(C, 1, figsize=(10, 6), sharex=True)
        if C == 1:  # 如果只有1通道，axes不是数组
            axes = [axes]

        for ch in range(C):
            axes[ch].plot(t, waves_np[i, ch])
            # axes[ch].set_ylabel("Amplitude")
            # axes[ch].legend(loc="upper right")
            axes[ch].grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

        # axes[-1].set_xlabel("Time (s)")
        # fig.suptitle(f"Sample {i}")
        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"sample_{i}.png"), dpi=200)
            plt.close(fig)