from tqdm import tqdm
from configs.tools import noise_alpha, cosine_similarity_batch
import torch
from configs.train_parameter import CFG

def train_fm_one_epoch(model, loader, fm, optimizer, scheduler, ema, device, epoch, writer, global_step):

    model.train()
    running_loss = running_cos = running_ratio = 0.0
    n_seen = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    alpha = noise_alpha(epoch)  # 噪声curriculum

    for spec, meta, fault, _, trace in pbar:
        spec = spec.to(device)
        meta = meta.to(device)
        fault = fault.to(device)

        # Flow Matching：缩放 z=alpha*z
        z = torch.randn_like(spec)
        z = alpha * z
        t, xt, ut = fm.sample_location_and_conditional_flow(z, spec)
        t, xt, ut = t.to(device), xt.to(device), ut.to(device)  # ut = x - alpha*z（RF/线性CFM）

        # 预测
        optimizer.zero_grad(set_to_none=True)
        vt = model(t, xt, meta, fault)

        # —— 预条件化 + 角度损失 ——
        if CFG.USE_PER_SAMPLE_RMS:
            with torch.no_grad():
                scale = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().clamp_min(1e-6)
        else:
            scale = 1

        diff = (vt - ut) / scale
        mse  = (diff * diff).mean()
        cos_b = cosine_similarity_batch(vt, ut)  # (B,)
        loss  = mse + CFG.ANGLE_LOSS_WEIGHT * (1.0 - cos_b.mean())

        loss.backward()

        # 是否梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()
        ema.update(model)

        # 监控量
        with torch.no_grad():
            v_norm = vt.view(vt.size(0), -1).norm(dim=1) + 1e-8
            u_norm = ut.view(ut.size(0), -1).norm(dim=1) + 1e-8
            ratio = (v_norm / u_norm).mean()

        bsz = spec.size(0)
        running_loss += loss.item() * bsz
        running_cos  += cos_b.mean().item() * bsz
        running_ratio+= ratio.item() * bsz
        n_seen += bsz

        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "cos": f"{cos_b.mean().item():.3f}",
                          "|v|/|u|": f"{ratio.item():.3f}",
                          "α": f"{alpha:.2f}"})

        if writer is not None:
            writer.add_scalar("train/step_loss", loss.item(), global_step)
            writer.add_scalar("train/step_cos",  cos_b.mean().item(), global_step)
            writer.add_scalar("train/step_ratio", ratio.item(), global_step)
        global_step += 1

    epoch_loss = running_loss / n_seen
    epoch_cos  = running_cos  / n_seen
    epoch_ratio= running_ratio/ n_seen

    if writer is not None:
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train/epoch_cos",  epoch_cos,  epoch)
        writer.add_scalar("train/epoch_ratio",epoch_ratio,epoch)
        writer.add_scalar("train/noise_alpha",alpha, epoch)

    return epoch_loss, epoch_cos, epoch_ratio, global_step