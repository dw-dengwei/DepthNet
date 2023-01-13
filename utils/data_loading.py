import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        depth_dir: str, 
        mask_dir: str,
        scale: float = 1.0, 
        depth_suffix: str = 'Range',
        mask_prefix: str = 'Lable'
    ):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.depth_suffix = depth_suffix
        self.mask_prefix = mask_prefix

        self.ids = [splitext(file)[0] for file in listdir(image_dir) if not file.startswith('.')]
        # self.ids = self.ids[: 100]
        if not self.ids:
            raise RuntimeError(f'No input file found in {image_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask, is_depth):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        if is_depth:
            pil_img = pil_img.convert('L')
        else:
            pil_img = pil_img.convert('RGB')
            
        img_ndarray = np.asarray(pil_img)


        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255.0

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.image_dir.glob(name + '.*'))
        depth_file = list(self.depth_dir.glob(name[:-8] + self.depth_suffix + '.*'))
        # print(self.mask_prefix + '_' + name, self.mask_dir)
        mask_file = list(self.mask_dir.glob(self.mask_prefix + '_' + name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(depth_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {depth_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {depth_file}'
        img = self.load(img_file[0])
        depth = self.load(depth_file[0])
        mask = self.load(mask_file[0])

        assert img.size == depth.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {depth.size}'

        img = self.preprocess(img, self.scale, is_mask=False, is_depth=False)
        depth = self.preprocess(depth, self.scale, is_mask=False, is_depth=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'depth': torch.as_tensor(depth.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, dir_mask, scale=1):
        super().__init__(images_dir, masks_dir, dir_mask, scale)
