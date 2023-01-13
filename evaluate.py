import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from utils.dice_score import multiclass_dice_coeff, dice_coeff

from loss import certirian
import torch.nn as nn


def mask_visual(label):
    label_2_color = [0x00, 0x64, 0xa9, 0xfe, 0xff, 0xaa]

    color = torch.zeros(label.size())
    for i in range(6):
        color[label == i] = label_2_color[i]
    color = color.unsqueeze(1)

    return color.repeat((1, 3, 1, 1))


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        images, depth_true, mask_true = batch['image'], batch['depth'], batch['mask']
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)

        recon_true = images.clone().detach()
        depth_true = depth_true.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # depth_true = depth_true.to(device=device, dtype=torch.long)
        # depth_true = F.one_hot(depth_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            recon_pred, \
            depth_pred, \
            mask_pred, \
            pred_recon_dis, \
            pred_depth_dis, \
            true_recon_dis, \
            true_depth_dis = net(images, recon_true, depth_true, mask_true)

            loss, \
            (loss_gen, loss_dis_adv), \
            (x1000_loss_depth, x100_loss_mask, loss_recon) = certirian(
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
                device
            )
            soft = nn.Softmax2d()
            mask_pred = soft(mask_pred).argmax(1)
            visual_pred = mask_visual(mask_pred)
            visual_true = mask_visual(mask_true)
            val = {
                'recon': {
                    'true': wandb.Image(recon_true.cpu()),
                    'pred': wandb.Image(recon_pred.cpu()),
                },
                'depth': {
                    'true': wandb.Image(depth_true.cpu()),
                    'pred': wandb.Image(depth_pred.cpu()),
                },
                'mask':{
                    'pred': wandb.Image(visual_pred.cpu(), mode='L'),
                    'true': wandb.Image(visual_true.cpu(), mode='L')
                },
            }

            # convert to one-hot format
            # if net.n_classes == 1:
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, depth_true, reduce_batch_first=False)
            # else:
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], depth_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    return loss, (loss_gen, loss_dis_adv), (x1000_loss_depth, x100_loss_mask, loss_recon), val
