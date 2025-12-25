import torch
from configs.train_parameter import CFG
from configs.tools import noise_alpha, cosine_similarity_batch

@torch.no_grad()
def evaluate(model, loader, fm,
             device: torch.device, epoch: int, writer):
    model.eval()

    running_loss = running_cos = running_ratio = 0.0
    n_seen = 0

    for spec, meta, fault, _, trace in loader:
        spec_std = spec.to(device)
        meta = meta.to(device)
        fault = fault.to(device)

        # 评估不做CFG dropout；噪声用完整 alpha=1
        z = torch.randn_like(spec_std)
        t, xt, ut = fm.sample_location_and_conditional_flow(z, spec_std)
        t, xt, ut = t.to(device), xt.to(device), ut.to(device)

        vt = model(t, xt, meta, fault)
        if CFG.USE_PER_SAMPLE_RMS:
            with torch.no_grad():
                scale = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().clamp_min(1e-6)
        else:
            scale = 1

        diff = (vt - ut) / scale
        mse  = (diff * diff).mean()
        cos_b = cosine_similarity_batch(vt, ut)
        loss = mse + CFG.ANGLE_LOSS_WEIGHT * (1.0 - cos_b.mean())

        with torch.no_grad():
            v_norm = vt.view(vt.size(0), -1).norm(dim=1) + 1e-8
            u_norm = ut.view(ut.size(0), -1).norm(dim=1) + 1e-8
            ratio = (v_norm / u_norm).mean()

        bsz = spec_std.size(0)
        running_loss += loss.item() * bsz
        running_cos  += cos_b.mean().item() * bsz
        running_ratio+= ratio.item() * bsz
        n_seen += bsz

    val_loss = running_loss / n_seen
    val_cos  = running_cos  / n_seen
    val_ratio= running_ratio/ n_seen

    if writer is not None:
        writer.add_scalar("val/loss",  val_loss,  epoch)
        writer.add_scalar("val/cos",   val_cos,   epoch)
        writer.add_scalar("val/ratio", val_ratio, epoch)

    return val_loss