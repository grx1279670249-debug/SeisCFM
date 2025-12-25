import os
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    SAVE_DIR = os.path.join("models", "Flow_Matching_3") # 保存路径
    os.makedirs(SAVE_DIR, exist_ok=True)
    BATCH_SIZE = 16 # 批次大小
    N_EPOCHS   = 300 # 最大训练轮数
    LEARNING_RATE = 3e-4 # 初始学习率
    WEIGHT_DECAY  = 1e-2
    WARMUP_STEPS  = 3000
    EMA_DECAY     = 0.999
    GRAD_CLIP_NORM = 1.0
    SIGMA = 0.01                    # Rectified/Linear CFM
    ANGLE_LOSS_WEIGHT = 0.10        # 角度项权重 λ
    USE_PER_SAMPLE_RMS = False       # True: 按样本RMS预条件化；False: 用全局vel_scale
    CFG_DROP_PROB = 0.10            # 训练时置空条件概率
    GUIDANCE_SCALE = 1.7            # 采样时的 CFG 指导强度
    CURRICULUM_EPOCHS = 15

CFG = TrainConfig()