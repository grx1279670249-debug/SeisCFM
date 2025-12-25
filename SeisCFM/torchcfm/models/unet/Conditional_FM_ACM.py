# Conditional_FM.py —— S0（全集训练）稳态版
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from math import cos, pi

from NGA_Dataset import NGADataset
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from model_unet import SeismicUNet  # 注意：UNet 的输出头已改为线性，无激活/Norm
# ===== 在文件顶部（imports 附近）补充 =====
import math
from pathlib import Path
try:
    import torchdiffeq  # type: ignore
except ImportError:
    torchdiffeq = None