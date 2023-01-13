import torch.nn.functional as F
import torch.nn as nn
import torch


def certirian(
    recon_true,
    depth_true,
    mask_true,
    recon_pred,
    depth_pred,
    mask_pred,
    pred_recon_dis,
    pred_depth_dis,
    true_recon_dis,
    true_depth_dis,
    device,
):
    ce_loss = nn.CrossEntropyLoss()

    loss_gen_adv = F.mse_loss(
        pred_recon_dis,
        torch.ones(pred_recon_dis.size(), device=device)
    ) + F.mse_loss(
        pred_depth_dis,
        torch.ones(pred_depth_dis.size(), device=device)
    )

    loss_dis_adv = F.mse_loss(
        true_recon_dis,
        torch.ones(true_recon_dis.size(), device=device)
    ) + F.mse_loss(
        pred_recon_dis,
        torch.zeros(pred_recon_dis.size(), device=device)
    ) + F.mse_loss(
        true_depth_dis,
        torch.ones(true_depth_dis.size(), device=device)
    ) + F.mse_loss(
        pred_depth_dis,
        torch.zeros(pred_depth_dis.size(), device=device)
    )

    loss_depth = F.l1_loss(depth_pred, depth_true)
    loss_mask = ce_loss(mask_pred, mask_true)
    loss_recon = F.l1_loss(recon_pred, recon_true)

    loss_gen = \
        100 * loss_depth + \
        100 * loss_mask + \
        1* loss_recon + \
        loss_gen_adv 
                
    loss = loss_gen + loss_dis_adv
    return loss, \
        (loss_gen.detach(), loss_dis_adv.detach()), \
        (100 * loss_depth.detach(), 100 * loss_mask.detach(), 1* loss_recon.detach())