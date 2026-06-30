import os
import json
from collections import namedtuple

import numpy as np
from PIL import Image
import torch.utils.data as data


CityscapesClass = namedtuple('CityscapesClass', [
    'name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'
])


class Cityscapes(data.Dataset):

    classes = [
        CityscapesClass('flat',         0,   0, 'flat',         0, False, False, (128,  64, 128)),
        CityscapesClass('construction', 1,   1, 'construction', 1, False, False, ( 70,  70,  70)),
        CityscapesClass('object',       2,   2, 'object',       2, False, False, (153, 153, 153)),
        CityscapesClass('nature',       3,   3, 'nature',       3, False, False, (107, 142,  35)),
        CityscapesClass('sky',          4,   4, 'sky',          4, False, False, ( 70, 130, 180)),
        CityscapesClass('human',        5,   5, 'human',        5, False, False, (220,  20,  60)),
        CityscapesClass('vehicle',      6,   6, 'vehicle',      6, False, False, (  0,   0, 142)),
        CityscapesClass('unlabeled',    7, 255, 'void',         7, False, True,  (  0,   0,   0)),
    ]

    # Build lookup arrays
    id_to_train_id = np.array([c.train_id for c in classes], dtype=np.uint8)
    train_id_to_color = np.array([c.color for c in classes if c.train_id != 255], dtype=np.uint8)

    def __init__(self, root, split='val', mode='gtFine', transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.transform = transform


        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, mode, split)
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError(f"Cityscapes data not found in {self.root}")

        self.images = []
        self.targets = []

        for city in sorted(os.listdir(self.images_dir)):
            img_folder = os.path.join(self.images_dir, city)

            if os.path.isdir(img_folder):
                tgt_folder = os.path.join(self.targets_dir, city)

                for fname in sorted(os.listdir(img_folder)):
                    if not fname.endswith('_leftImg8bit.png'):
                        continue

                    img_path = os.path.join(img_folder, fname)
                    label_name = fname.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    tgt_path = os.path.join(tgt_folder, label_name)

                    if not os.path.isfile(tgt_path):
                        raise RuntimeError(f"Missing label: {tgt_path}")

                    self.images.append(img_path)
                    self.targets.append(tgt_path)

            else:
                fname = city

                if not fname.endswith('_leftImg8bit.png'):
                    continue

                img_path = img_folder
                label_name = fname.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                tgt_path = os.path.join(self.targets_dir, label_name)

                if not os.path.isfile(tgt_path):
                    raise RuntimeError(f"Missing label: {tgt_path}")

                self.images.append(img_path)
                self.targets.append(tgt_path)

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, target):
        arr = np.array(target, dtype=np.uint8)
        arr = cls.id_to_train_id[arr]
        return arr

    @classmethod
    def decode_target(cls, mask):
        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for tid, col in enumerate(cls.train_id_to_color):
            out[mask == tid] = col
        return out

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        tgt = Image.open(self.targets[index])
        if self.transform:
            img, tgt = self.transform(img, tgt)
        mask = self.encode_target(tgt)
        return img, mask

    @staticmethod
    def get_palette():
        return Cityscapes.train_id_to_color
