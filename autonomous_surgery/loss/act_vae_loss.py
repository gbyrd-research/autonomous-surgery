# autonomous_surgery/loss/act_vae_loss.py
# -*- coding: utf-8 -*-
"""
ACT/DETR-VAE loss for chunk supervision (classic ACT-style KL).

- recon: L1 or SmoothL1 over actions, masked by ~is_pad
- KL: KL(q(z|x) || N(0, I)) using posterior (mu, logvar)
  NOTE: Even if preds provides prior_mu/prior_logvar, we IGNORE them on purpose
        to match classic ACT behavior and avoid unstable learned-prior issues.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os
import torch
import torch.nn.functional as F

from autonomous_surgery.models.representation.representation_act_actor import ActOutput

Tensor = torch.Tensor


def _masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    x = x * mask.to(dtype=x.dtype)
    denom = mask.to(dtype=x.dtype).sum().clamp_min(eps) * x.shape[-1]   # <-- include action dimension
    return x.sum() / denom


def kl_q_to_std_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) )
    Returns scalar averaged over batch.
    """
    # this will prevent KL collapse, although it's unclear how this will affect
    # training
    free_bits = 0.1
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    klds = torch.clamp(klds, min=free_bits) # Mitigate posterior collapse
    kl = klds.sum(dim=1).mean()
    return kl

    # # klds: [B,Z]
    # klds = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    # return klds.sum(dim=1).mean()


def act_vae_loss(
    preds: ActOutput,
    gt_actions_norm: Tensor,
    is_pad: Tensor,
    kl_weight: float = 0.01,
    reduction: str = "mean",
) -> Tuple[Tensor, Dict[str, Tensor]]:

    b, k, a = gt_actions_norm.shape

    valid = ~is_pad.to(device=gt_actions_norm.device)

    # reconstruction loss
    recon_all = F.l1_loss(preds.actions_norm, gt_actions_norm, reduction="none")

    valid_3 = valid.unsqueeze(-1)  # [B,K,1]

    if reduction == "mean":
        recon = _masked_mean(recon_all, valid_3)
    elif reduction == "sum":
        recon = (recon_all * valid_3.to(recon_all.dtype)).sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # KL (classic ACT): ALWAYS KL(q || N(0, I)) using posterior only
    if (preds.mu is not None 
        and preds.logvar is not None 
        and preds.prior_mu is not None 
        and preds.prior_logvar is not None
    ):
        with torch.autocast("cuda", enabled=False):
            kl = kl_q_to_std_normal(preds.mu.float(), preds.logvar.float())

        loss = recon + float(kl_weight) * kl
        
        loss_dict: Dict[str, Tensor] = {
            "loss": loss.detach(),
            "recon": recon.detach(),
            "kl": kl.detach(),
        }

    else:
        loss = recon

        loss_dict: Dict[str, Tensor] = {
            "loss": loss.detach(),
            "recon": recon.detach(),
        }

    return loss, loss_dict

    # optional pad head loss
    # pad_loss = torch.zeros((), device=gt_actions_norm.device, dtype=gt_actions_norm.dtype)
    # if include_is_pad_loss and (is_pad is not None):
    #     if is_pad_hat is None:
    #         raise ValueError("include_is_pad_loss=True but preds has no is_pad_hat.")
    #     if is_pad_hat.shape != (b, k):
    #         raise ValueError(f"is_pad_hat must be [B,K]={b,k}, got {tuple(is_pad_hat.shape)}")
    #     target = is_pad.to(dtype=gt_actions_norm.dtype, device=gt_actions_norm.device)  # 1 for padded
    #     pad_all = F.binary_cross_entropy_with_logits(is_pad_hat, target, reduction="none")  # [B,K]
    #     pad_loss = pad_all.mean() if reduction == "mean" else pad_all.sum()

    #  + float(pad_loss_weight) * pad_loss

    # loss_dict: Dict[str, Tensor] = {
    #     "loss": loss.detach(),
    #     "recon": recon.detach(),
    #     "kl": kl.detach(),
    # }
    # if include_is_pad_loss:
    #     loss_dict["pad_bce"] = pad_loss.detach()

    # # ----------------------------
    # # DEBUG (prints rarely)
    # # ----------------------------
    # # ACT_VAE_DEBUG=1 开启；ACT_VAE_DEBUG_EVERY=200 控制频率
    # if os.environ.get("ACT_VAE_DEBUG", "0") == "1":
    #     every = int(os.environ.get("ACT_VAE_DEBUG_EVERY", "200"))
    #     if not hasattr(act_vae_loss, "_dbg_step"):
    #         act_vae_loss._dbg_step = 0
    #     act_vae_loss._dbg_step += 1

    #     if (act_vae_loss._dbg_step % every) == 0:
    #         with torch.no_grad():
    #             recon_v = float(recon.detach().cpu())
    #             kl_v = float(kl.detach().cpu())
    #             loss_v = float(loss.detach().cpu())
    #             pad_v = float(pad_loss.detach().cpu()) if include_is_pad_loss else 0.0

    #             has_post = (mu is not None) and (logvar is not None)
    #             has_prior = (prior_mu is not None) and (prior_logvar is not None)

    #             if has_post:
    #                 # raw term should be <= 0 ; expected KL should be >= 0
    #                 raw = (1.0 + logvar - mu.pow(2) - logvar.exp())     # [B,Z]
    #                 raw_sum = float(raw.sum(dim=1).mean().cpu())

    #                 kl_expected = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B,Z]
    #                 kl_expected_sum = float(kl_expected.sum(dim=1).mean().cpu())

    #                 mu_min, mu_max, mu_mean = float(mu.min().cpu()), float(mu.max().cpu()), float(mu.mean().cpu())
    #                 lv_min, lv_max, lv_mean = float(logvar.min().cpu()), float(logvar.max().cpu()), float(logvar.mean().cpu())
    #                 var = logvar.exp()
    #                 var_min, var_max, var_mean = float(var.min().cpu()), float(var.max().cpu()), float(var.mean().cpu())
    #             else:
    #                 raw_sum = 0.0
    #                 kl_expected_sum = 0.0
    #                 mu_min = mu_max = mu_mean = float("nan")
    #                 lv_min = lv_max = lv_mean = float("nan")
    #                 var_min = var_max = var_mean = float("nan")

    #             # Prior stats (ONLY for debugging visibility; NOT used in loss)
    #             if has_prior:
    #                 pm, plv = prior_mu, prior_logvar
    #                 pm_shape = tuple(pm.shape)
    #                 plv_shape = tuple(plv.shape)
    #                 prior_finite = bool(torch.isfinite(pm).all() and torch.isfinite(plv).all())

    #                 pm_min, pm_max, pm_mean = float(pm.min().cpu()), float(pm.max().cpu()), float(pm.mean().cpu())
    #                 plv_min, plv_max, plv_mean = float(plv.min().cpu()), float(plv.max().cpu()), float(plv.mean().cpu())

    #                 too_small = float((plv < -80).float().mean().cpu() * 100.0)
    #                 too_large = float((plv > 80).float().mean().cpu() * 100.0)
    #             else:
    #                 pm_shape = plv_shape = None
    #                 prior_finite = True
    #                 pm_min = pm_max = pm_mean = float("nan")
    #                 plv_min = plv_max = plv_mean = float("nan")
    #                 too_small = too_large = 0.0

    #             print(
    #                 "[ACT_VAE_LOSS DEBUG]"
    #                 f" step={act_vae_loss._dbg_step}"
    #                 f" recon={recon_v:.6f}"
    #                 f" kl={kl_v:.6f}"
    #                 f" (kl_w={float(kl_weight):.6g})"
    #                 f" pad={pad_v:.6f}"
    #                 f" loss={loss_v:.6f}"
    #                 f" post={has_post} prior_present={has_prior}(ignored)"
    #                 f" raw_sum={raw_sum:.6f}"
    #                 f" kl_expected={kl_expected_sum:.6f}"
    #                 f" mu[min,max,mean]=[{mu_min:.3g},{mu_max:.3g},{mu_mean:.3g}]"
    #                 f" logvar[min,max,mean]=[{lv_min:.3g},{lv_max:.3g},{lv_mean:.3g}]"
    #                 f" var[min,max,mean]=[{var_min:.3g},{var_max:.3g},{var_mean:.3g}]"
    #                 f" prior_finite={prior_finite}"
    #                 f" prior_mu_shape={pm_shape} prior_logvar_shape={plv_shape}"
    #                 f" prior_mu[min,max,mean]=[{pm_min:.3g},{pm_max:.3g},{pm_mean:.3g}]"
    #                 f" prior_logvar[min,max,mean]=[{plv_min:.3g},{plv_max:.3g},{plv_mean:.3g}]"
    #                 f" prior_logvar_pct(<-80)={too_small:.2f}%"
    #                 f" prior_logvar_pct(>80)={too_large:.2f}%"
    #             )

    # return loss, loss_dict