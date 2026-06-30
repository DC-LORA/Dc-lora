import os
import json
from collections import namedtuple

import numpy as np
from PIL import Image
import torch.utils.data as data


class GTA5Class(namedtuple('GTA5Class', ['name','id','train_id','category','category_id','has_instances','ignore_in_eval','color'])):
    """Metadata for each class."""
    pass


class GTA5(data.Dataset):

    classes = [
        GTA5Class('flat',         0,   0, 'flat',         0, False, False, (128,  64, 128)),
        GTA5Class('construction', 1,   1, 'construction', 1, False, False, ( 70,  70,  70)),
        GTA5Class('object',       2,   2, 'object',       2, False, False, (153, 153, 153)),
        GTA5Class('nature',       3,   3, 'nature',       3, False, False, (107, 142,  35)),
        GTA5Class('sky',          4,   4, 'sky',          4, False, False, ( 70, 130, 180)),
        GTA5Class('human',        5,   5, 'human',        5, False, False, (220,  20,  60)),
        GTA5Class('vehicle',      6,   6, 'vehicle',      6, False, False, (  0,   0, 142)),
        GTA5Class('unlabeled',    7, 255, 'void',         7, False, True,  (  0,   0,   0)),
    ]

    id_to_train_id    = np.array([c.train_id for c in classes], dtype=np.uint8)
    train_id_to_color = np.array([c.color    for c in classes if c.train_id != 255], dtype=np.uint8)

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'images')
        self.labels_dir = os.path.join(self.root, 'labels')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
            raise RuntimeError(
                f"GTA5 folders not found under {self.root}."
            )

        self.images = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ])

        self.labels = []
        for img_path in self.images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, f"{stem}.png")
            if not os.path.isfile(label_path):
                raise RuntimeError(f"Label file not found for image {stem}: {label_path}")
            self.labels.append(label_path)

        if len(self.images) != len(self.labels):
            raise RuntimeError("GTA5: image and label count mismatch")

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, label):
        """Map raw labelIds -> trainIds 0–6, 255."""
        arr = np.array(label, dtype=np.uint8)
        return cls.id_to_train_id[arr]

    @classmethod
    def decode_target(cls, mask):
        """Map trainIds 0–6 -> RGB for visualization."""
        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for tid, col in enumerate(cls.train_id_to_color):
            out[mask == tid] = col
        return out

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lbl = Image.open(self.labels[idx])
        if self.transform:
            img, lbl = self.transform(img, lbl)
        mask = self.encode_target(lbl)
        return img, mask

    @staticmethod
    def get_palette():
        return GTA5.train_id_to_color
