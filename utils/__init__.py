from .utils import *
from .visualizer import Visualizer
from .scheduler import PolyLR
from .loss import FocalLoss

import torch
import copy

class EMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                model_v = msd[k].detach()
                if model_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                else:
                    ema_v.copy_(model_v)

    def to(self, device):
        self.ema = self.ema.to(device)


