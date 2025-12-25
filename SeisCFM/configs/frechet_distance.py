import numpy as np
from scipy import linalg

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-8):
    """
    经典 FID / Fréchet distance 实现。
    mu1, mu2: 均值向量 (D,)
    sigma1, sigma2: 协方差矩阵 (D, D)
    返回：标量 FD 值
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 协方差矩阵乘积的矩阵平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        # 数值不稳定时加一点对角线扰动
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # 理论上结果应为实数，数值误差可能产生极小虚部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fd)

def frechet_from_embeddings(emb_real, emb_fake):
    """
    emb_*: np.ndarray, shape (N, D)
    """
    mu_real = np.mean(emb_real, axis=0)
    mu_fake = np.mean(emb_fake, axis=0)
    sigma_real = np.cov(emb_real, rowvar=False)
    sigma_fake = np.cov(emb_fake, rowvar=False)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

