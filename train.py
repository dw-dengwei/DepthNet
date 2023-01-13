import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

from loss import certirian

dir_img = Path('input_image')
dir_mask = Path('input_mask')
dir_depth = Path('input_depth')
dir_checkpoint = Path('./checkpoints/')

def mask_visual(label):
    label_2_color = [0x00, 0x64, 0xa9, 0xfe, 0xff, 0xaa]

    color = torch.zeros(label.size())
    for i in range(6):
        color[label == i] = label_2_color[i]
    color = color.unsqueeze(1)

    return color.repeat((1, 3, 1, 1))


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 32,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_depth, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_depth, dir_mask, img_scale, depth_suffix='Range')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                depth_true = batch['depth']
                mask_true = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                recon_true = images.clone().detach()
                depth_true = depth_true.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
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
                

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train': {
                        'loss': loss.item(),
                        'loss_gen': loss_gen.item(),
                        'loss_dis_adv': loss_dis_adv.item(),
                        'x1000_loss_depth': x1000_loss_depth.item(),
                        'x100_loss_mask': x100_loss_mask.item(), 
                        'loss_recon': loss_recon.item(),
                        'pred_recon_dis': pred_recon_dis.mean().item(),
                        'true_recon_dis': true_recon_dis.mean().item(),
                        'pred_depth_dis': pred_depth_dis.mean().item(),
                        'true_depth_dis': true_depth_dis.mean().item(),
                    },
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, \
                        (val_loss_gen, val_loss_dis_adv), \
                        (val_x1000_loss_depth, val_x100_loss_mask, val_loss_recon), \
                        val_log \
                            = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        soft = nn.Softmax2d()
                        mask_pred = soft(mask_pred).argmax(1)
                        visual_pred = mask_visual(mask_pred)
                        visual_true = mask_visual(mask_true)
                        norm_recon_pred = np.zeros(recon_pred.size(), dtype=np.float32)
                        cv2.normalize(
                            recon_pred.detach().cpu().numpy(), 
                            norm_recon_pred,  
                            alpha=0, 
                            beta=1, 
                            norm_type=cv2.NORM_MINMAX
                        )
                        norm_recon_pred *= 255

                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation loss': val_score,
                            'recon': {
                                'true': wandb.Image(recon_true.cpu()),
                                # 'pred': wandb.Image(recon_pred.cpu()),
                                'pred': wandb.Image(torch.tensor(norm_recon_pred)),
                            },
                            'depth': {
                                'true': wandb.Image(depth_true.cpu()),
                                'pred': wandb.Image(depth_pred.cpu()),
                            },
                            'mask':{
                                'pred': wandb.Image(visual_pred.cpu(), mode='L'),
                                'true': wandb.Image(visual_true.cpu(), mode='L')
                            },
                            'val': val_log,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    wandb.login(key='253339369d60f859c4c8e26ab213edc19e5c5b0d')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
