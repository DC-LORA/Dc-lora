from __future__ import annotations

import argparse
import copy
import hashlib
import math
import pickle
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.func import functional_call, jvp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TVT
from torchvision.transforms import functional as TF
from tqdm import tqdm


DATASET_ALIASES = {
    "CS": ("cityscapes", "citysscapes"),
    "IDD": ("idd",),
    "MAP": ("mapillary",),
}
METHODS = ("dtst", "csfda", "stable-st")
TARGETS = {"cityscapes": "CS", "idd": "IDD", "mapillary": "MAP"}


def save_ckpt(path, student, teacher, step, score):
    teacher_state = teacher.state_dict()
    torch.save({
        "model_state": student.state_dict(),
        "teacher_state": teacher_state,
        "ema_state_merged": teacher_state,
        "iteration": step,
        "best_mIoU": score,
    }, path)


class TargetImages(Dataset):
    def __init__(self, domain, min_samples=0):
        self.root, self.images = scan_target_images(domain)
        if min_samples:
            self.images *= math.ceil(min_samples / len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(TRAIN_HW[::-1], Image.BILINEAR)
        return pil_to_normalized(image), path.relative_to(self.root).as_posix()


def fast_hist(labels, predictions, count):
    valid = (labels >= 0) & (labels < count)
    indices = count * labels[valid].astype(np.int64) + predictions[valid]
    return np.bincount(indices, minlength=count * count).reshape(count, count)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        intersection = np.diag(hist)
        return intersection / (hist.sum(0) + hist.sum(1) - intersection)


def adjust_learning_rate(optimizer, step, steps):
    lr = poly_lr(step, steps)
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = HEAD_LR_MULT * lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for image, label in loader:
        logits = model(image.to(device, non_blocking=True))
        logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=True)
        for target, pred in zip(label.numpy(), logits.argmax(1).cpu().numpy()):
            hist += fast_hist(target.ravel(), pred.ravel(), NUM_CLASSES)
    iou = per_class_iu(hist) * 100.0
    return {"mIoU": float(np.nanmean(iou)), "class_iou": dict(zip(CLASS_NAMES, iou.tolist()))}


NUM_CLASSES = 7
OUTPUT_STRIDE = 16
CLASS_NAMES = ["flat", "construction", "object", "nature", "sky", "human", "vehicle"]


TRAIN_HW = (320, 640)
FULL_HW = (480, 960)
VAL_HW = (320, 640)
BATCH_SIZE = 2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IGNORE = 255


BASE_LR = 2.5e-4
HEAD_LR_MULT = 10.0
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
POLY_POWER = 0.9
OFFICIAL_LR_MAX_ITER = 62_500


INPUT_SCALE_MIN = 0.75
INPUT_SCALE_MAX = 1.25
HFLIP_PROB = 0.5
JITTER_BRIGHTNESS = 0.5
JITTER_CONTRAST = 0.5
JITTER_SATURATION = 0.5
JITTER_HUE = 0.2
BLUR_PROB = 0.5


DTU_QUERY_START = 10
DTU_QUERY_STEP = 5
DTU_META_MAX_UPDATE = 21
DTU_EMA_WEIGHT = 0.99
NFD_STYLE_KEEP = 0.125


TCR_TOPK_CANDIDATE = 0.50
TCR_UPDATE_FREQUENCY = 3000
META_TOP_K = 0.05


THRESHOLD_BETA = 0.001
RUN_CONF_UPPER = 0.80
RUN_CONF_LOWER = 0.20
STAT_MOMENTUM = 0.90


TCR_OPEN = True
METAPL_OPEN = True
METAPL_VAL_UPDATE = 3000
CANDIDATE_INIT_HW = (256, 512)
CONF_HIST_BINS = 65536
MIX_CLASSES_7 = [1, 2, 5, 6]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def looks_like_label(p: Path) -> bool:
    s = p.stem.lower()
    markers = (
        "gtfine", "labelid", "trainid", "_label", "_mask", "annotation",
        "semantic", "_color"
    )
    return any(x in s for x in markers) and "leftimg8bit" not in s


def resolve_dataset_root(domain: str) -> Path:
    folders = {path.name.lower(): path for path in DATA_ROOT.iterdir() if path.is_dir()}
    for alias in DATASET_ALIASES[domain]:
        if alias in folders:
            return folders[alias]
    raise FileNotFoundError(DATA_ROOT / DATASET_ALIASES[domain][0])


def find_split_dir(root: Path, split: str, kind: str) -> Path:
    if kind == "images":
        candidates = [
            root / "leftImg8bit" / split,
            root / split / "leftImg8bit",
            root / "images" / split,
            root / split / "images",
            root / "rgb" / split,
            root / split / "rgb",
            root / split,
        ]
    else:
        candidates = [
            root / "gtFine" / split,
            root / split / "gtFine",
            root / "labels" / split,
            root / split / "labels",
            root / "annotations" / split,
            root / split / "annotations",
            root / split,
        ]

    for c in candidates:
        if not c.is_dir():
            continue
        files = [p for p in c.rglob("*") if p.is_file() and is_image_file(p)]
        if kind == "images":
            good = [p for p in files if not looks_like_label(p)]
        else:
            good = [p for p in files if looks_like_label(p)]
        if good:
            return c
    raise FileNotFoundError(f"Cannot locate {kind} for {root} split={split}")


def image_key(stem: str) -> str:
    s = stem
    for suffix in ("_leftImg8bit", "_rgb", "_image", "_img"):
        if s.lower().endswith(suffix.lower()):
            s = s[:-len(suffix)]
    return s.lower()


def label_key(stem: str) -> str:
    s = stem
    suffixes = (
        "_gtFine_labelIds", "_gtFine_labelTrainIds", "_gtFine_trainIds",
        "_gtFine_instanceIds", "_gtFine_color", "_labelIds", "_trainIds",
        "_label", "_labels", "_mask", "_annotation", "_semantic",
    )
    for suffix in suffixes:
        if s.lower().endswith(suffix.lower()):
            s = s[:-len(suffix)]
            break
    return s.lower()


def label_priority(p: Path) -> int:
    s = p.stem.lower()
    if "labelids" in s:
        return 0
    if "labeltrainids" in s or "trainids" in s:
        return 1
    if "_label" in s:
        return 2
    if "_mask" in s:
        return 3
    if "color" in s:
        return 99
    return 10


def sample_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    h = hashlib.sha1(rel.lower().encode("utf-8")).hexdigest()[:10]
    stem = path.stem.replace("_leftImg8bit", "")
    return f"{stem}_{h}"


def scan_target_images(domain: str, split: str = "train") -> Tuple[Path, List[Path]]:
    root = resolve_dataset_root(domain)
    img_dir = find_split_dir(root, split, "images")
    images = sorted(
        p for p in img_dir.rglob("*")
        if p.is_file() and is_image_file(p) and not looks_like_label(p)
    )
    return root, images


def scan_val_pairs(domain: str, split: str = "val") -> Tuple[Path, List[Tuple[Path, Path]]]:
    root = resolve_dataset_root(domain)
    img_dir = find_split_dir(root, split, "images")
    lab_dir = find_split_dir(root, split, "labels")

    images = sorted(
        p for p in img_dir.rglob("*")
        if p.is_file() and is_image_file(p) and not looks_like_label(p)
    )
    labels = [
        p for p in lab_dir.rglob("*")
        if p.is_file() and is_image_file(p) and looks_like_label(p)
    ]

    by_key: Dict[str, Path] = {}
    for p in sorted(labels, key=lambda q: (label_priority(q), str(q))):
        by_key.setdefault(label_key(p.stem), p)

    pairs = [(image, by_key[image_key(image.stem)]) for image in images]
    return root, pairs


def pil_to_normalized(im: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(im)
    return TF.normalize(x, MEAN, STD)


_STABLE_COLOR = TVT.ColorJitter(
    brightness=JITTER_BRIGHTNESS,
    contrast=JITTER_CONTRAST,
    saturation=JITTER_SATURATION,
    hue=JITTER_HUE,
)
_STABLE_BLUR = TVT.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))


@dataclass
class TargetIndex:
    domain: str
    root: Path
    ids: List[str]
    paths: List[Path]
    id_to_path: Dict[str, Path]


def make_target_index(domain: str) -> TargetIndex:
    root, paths = scan_target_images(domain, "train")
    ids = [sample_id(p, root) for p in paths]
    return TargetIndex(domain, root, ids, paths, dict(zip(ids, paths)))


def official_target_weak_view(clean_full: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    image = clean_full
    if random.random() < BLUR_PROB:
        image = _STABLE_BLUR(image)
    if random.random() < 0.9:
        image = _STABLE_COLOR(image)
    else:
        image = TF.rgb_to_grayscale(image, num_output_channels=3)

    full_w, full_h = image.size
    scale = INPUT_SCALE_MIN + (INPUT_SCALE_MAX - INPUT_SCALE_MIN) * random.random()
    sh, sw = int(full_h * scale), int(full_w * scale)
    image = image.resize((sw, sh), Image.BICUBIC)

    crop_h, crop_w = TRAIN_HW
    if sw < crop_w or sh < crop_h:
        pad_r = max(crop_w - sw, 0)
        pad_b = max(crop_h - sh, 0)
        image = TF.pad(image, (0, 0, pad_r, pad_b), fill=0)
        sh, sw = image.size[1], image.size[0]

    y1 = random.randint(0, sh - crop_h)
    x1 = random.randint(0, sw - crop_w)
    y2, x2 = y1 + crop_h, x1 + crop_w
    image = TF.crop(image, y1, x1, crop_h, crop_w)

    do_flip = random.random() < HFLIP_PROB
    if do_flip:
        image = TF.hflip(image)

    geom = torch.tensor([sh, sw, y1, y2, x1, x2, int(do_flip)], dtype=torch.int64)
    return pil_to_normalized(image), geom


def full2weak_dynamic(feat: torch.Tensor, geom: torch.Tensor, full_hw: Tuple[int, int] = FULL_HW) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError(f"full2weak expects BxCxHxW, got {tuple(feat.shape)}")
    full_h, full_w = full_hw
    fh, fw = feat.shape[-2:]
    sy, sx = fh / float(full_h), fw / float(full_w)
    out = []
    for b in range(feat.shape[0]):
        sh, sw, y1, y2, x1, x2, flip = [int(v) for v in geom[b].tolist()]
        rh = max(1, int(round(sh * sy)))
        rw = max(1, int(round(sw * sx)))
        z = F.interpolate(feat[b:b+1], size=(rh, rw), mode="bilinear", align_corners=True)
        fy1 = int(round(y1 * sy))
        fy2 = int(round(y2 * sy))
        fx1 = int(round(x1 * sx))
        fx2 = int(round(x2 * sx))
        fy2 = max(fy1 + 1, fy2)
        fx2 = max(fx1 + 1, fx2)
        z = z[:, :, fy1:fy2, fx1:fx2]
        expected_h = max(1, int(round(TRAIN_HW[0] * sy)))
        expected_w = max(1, int(round(TRAIN_HW[1] * sx)))
        if z.shape[-2:] != (expected_h, expected_w):
            z = F.interpolate(z, size=(expected_h, expected_w), mode="bilinear", align_corners=True)
        if flip:
            z = torch.flip(z, dims=[3])
        out.append(z)
    return torch.cat(out, 0)


def full_logits_to_weak(logits_full: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    out = []
    for b in range(logits_full.shape[0]):
        sh, sw, y1, y2, x1, x2, flip = [int(v) for v in geom[b].tolist()]
        z = F.interpolate(logits_full[b:b+1], size=(sh, sw), mode="bilinear", align_corners=True)
        z = z[:, :, y1:y2, x1:x2]
        if z.shape[-2:] != TRAIN_HW:
            z = F.interpolate(z, size=TRAIN_HW, mode="bilinear", align_corners=True)
        if flip:
            z = torch.flip(z, dims=[3])
        out.append(z)
    return torch.cat(out, 0)


class ValidationDataset(Dataset):
    def __init__(self, domain):
        _, self.pairs = scan_val_pairs(domain)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(VAL_HW[::-1], Image.BILINEAR)
        with Image.open(label_path) as label:
            gt = np.asarray(label.resize(VAL_HW[::-1], Image.NEAREST), dtype=np.int64).copy()
        gt[gt == 7] = IGNORE
        valid = (gt == IGNORE) | ((gt >= 0) & (gt < NUM_CLASSES))
        if gt.ndim != 2 or not valid.all():
            raise ValueError(f"Expected a seven-class ID mask: {label_path}")
        return pil_to_normalized(image), torch.from_numpy(gt)


def build_full_model(pretrained_backbone=True):
    from network.modeling import deeplabv2_resnet101

    return deeplabv2_resnet101(num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE,
                              pretrained_backbone=pretrained_backbone)


def clean_dlv2_keys(state):
    out = OrderedDict()
    for key, value in state.items():
        for prefix in ("module.", "base_model.model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out


def select_state(checkpoint):
    for key in ("ema_state_merged", "teacher_state", "model_state_merged", "model_state", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_full_model(path, device, pretrained_backbone=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_full_model(pretrained_backbone)
    if "feature_extractor" in checkpoint:
        model.backbone.load_state_dict({
            key.removeprefix("backbone."): value
            for key, value in checkpoint["feature_extractor"].items()
        })
        model.classifier.load_state_dict({
            key.removeprefix("classifier."): value
            for key, value in checkpoint["classifier"].items()
        })
    else:
        model.load_state_dict(clean_dlv2_keys(select_state(checkpoint)))
    return model.to(device)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone(x)
        return features, features


class Classifier(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, features, size=None):
        logits = self.classifier(features)
        if size is not None and logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=True)
        return logits


def configure_csfda_bn_only(model):
    model.requires_grad_(False)
    trainable = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            trainable.extend(module.parameters())
    return trainable


@torch.no_grad()
def csfda_ema(teacher, student, eta):

    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(eta).add_(sp.data, alpha=1.0 - eta)


    s_buf = dict(student.named_buffers())
    for name, tb in teacher.named_buffers():
        if name not in s_buf:
            continue
        sb = s_buf[name]
        if torch.is_floating_point(tb):
            tb.data.mul_(eta).add_(sb.data, alpha=1.0 - eta)
        else:
            tb.data.copy_(sb.data)


def csfda_colorjitter_gaussian(x, args):
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    rgb = (x * std + mean).clamp(0, 1)
    ops = []
    if args.csfda_brightness > 0:
        b = random.uniform(max(0.0, 1.0 - args.csfda_brightness),
                           1.0 + args.csfda_brightness)
        ops.append(("b", b))
    if args.csfda_contrast > 0:
        c = random.uniform(max(0.0, 1.0 - args.csfda_contrast),
                           1.0 + args.csfda_contrast)
        ops.append(("c", c))
    if args.csfda_saturation > 0:
        s = random.uniform(max(0.0, 1.0 - args.csfda_saturation),
                           1.0 + args.csfda_saturation)
        ops.append(("s", s))
    if args.csfda_hue > 0:
        h = random.uniform(-args.csfda_hue, args.csfda_hue)
        ops.append(("h", h))
    random.shuffle(ops)

    for kind, val in ops:
        if kind == "b":
            rgb = TF.adjust_brightness(rgb, val)
        elif kind == "c":
            rgb = TF.adjust_contrast(rgb, val)
        elif kind == "s":
            rgb = TF.adjust_saturation(rgb, val)
        elif kind == "h":
            rgb = TF.adjust_hue(rgb, val)

    if args.csfda_noise_std > 0:
        rgb = rgb + torch.randn_like(rgb) * args.csfda_noise_std

    return (rgb.clamp(0, 1) - mean) / std


@torch.no_grad()
def csfda_pseudo_and_reliability(teacher, x, args):
    clean_logits = teacher(x)
    clean_logits = F.interpolate(
        clean_logits, size=x.shape[-2:], mode="bilinear", align_corners=True
    )
    clean_prob = F.softmax(clean_logits, dim=1)
    confidence, pseudo = clean_prob.max(dim=1)

    aug_conf = []
    for _ in range(args.csfda_views):
        xa = csfda_colorjitter_gaussian(x, args)
        la = teacher(xa)
        la = F.interpolate(la, size=x.shape[-2:], mode="bilinear", align_corners=True)
        pa = F.softmax(la, dim=1)
        aug_conf.append(pa.max(dim=1).values)

    uncertainty = torch.stack(aug_conf, dim=0).std(dim=0, unbiased=False)

    reliable = torch.zeros_like(pseudo, dtype=torch.bool)
    q = float(args.csfda_percentile) / 100.0

    for k in range(NUM_CLASSES):
        mk = pseudo == k
        if not mk.any():
            continue
        tc = torch.quantile(confidence[mk], q)
        tu = torch.quantile(uncertainty[mk], q)
        reliable |= mk & (confidence >= tc) & (uncertainty <= tu)

    return pseudo, reliable


def csfda_step(student, teacher, x, optimizer, state, args):
    teacher.eval()
    with torch.no_grad():
        pseudo, reliable = csfda_pseudo_and_reliability(teacher, x, args)

    student.train()
    logits = student(x)
    logits = F.interpolate(
        logits, size=x.shape[-2:], mode="bilinear", align_corners=True
    )

    ce_map = F.cross_entropy(logits, pseudo, reduction="none")
    if reliable.any():
        loss_ce = ce_map[reliable].mean()
    else:
        loss_ce = logits.sum() * 0.0

    prob = F.softmax(logits, dim=1)
    loss_ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()


    loss = loss_ce + state["mu_e"] * loss_ent

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    csfda_ema(teacher, student, args.csfda_eta)


    state["mu_e"] *= math.exp(-args.csfda_beta)

    return loss.item()


@torch.no_grad()
def dtst_update_running_conf(probs, running_conf, args):
    probs_avg = probs.mean(dim=(0, 2, 3))
    update_index = probs_avg > 1e-8
    new_index = update_index & (running_conf == args.dtst_threshold_beta)
    running_conf[new_index] = probs_avg[new_index]
    running_conf.mul_(args.dtst_stat_momentum).add_(
        probs_avg, alpha=1.0 - args.dtst_stat_momentum
    )


@torch.no_grad()
def dtst_positive_pseudo(probs, running_conf, args):
    B, C, H, W = probs.shape
    max_conf, max_idx = probs.max(dim=1)

    peaks = torch.zeros((B, C), device=probs.device, dtype=probs.dtype)
    for c in range(C):
        pc = torch.where(max_idx == c, max_conf, torch.zeros_like(max_conf))
        peaks[:, c] = pc.flatten(1).max(dim=1).values

    peaks = peaks * args.dtst_run_conf_upper
    discount = 1.0 - torch.exp(
        -running_conf / float(args.dtst_threshold_beta)
    )
    peaks = peaks * discount.view(1, C)
    peaks = peaks.clamp_min(args.dtst_run_conf_lower)

    pix_thr = torch.gather(peaks, 1, max_idx.flatten(1)).view(B, H, W)
    pseudo = max_idx.clone()
    pseudo[max_conf <= pix_thr] = IGNORE
    return pseudo


def dtst_negative_bce(student_logits, teacher_probs):
    neg_valid = teacher_probs <= (1.0 / float(NUM_CLASSES))
    if not neg_valid.any():
        return student_logits.sum() * 0.0
    zeros = torch.zeros_like(student_logits)
    bce = F.binary_cross_entropy_with_logits(student_logits, zeros, reduction="none")
    return bce[neg_valid].mean()


@torch.no_grad()
def dtst_ent_proxy(model, x):
    logits = model(x)
    logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
    pmax = F.softmax(logits, dim=1).max(dim=1).values.clamp_min(1e-8)

    score = 1.0 - (-pmax * torch.log2(pmax) * 2.0)
    return float(score.mean().item())


@torch.no_grad()
def dtst_weight_ema(history_teacher, student, alpha):
    for hp, sp in zip(history_teacher.parameters(), student.parameters()):
        hp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)


def dtst_dynamic_teacher_update(history_teacher, student, state, args, device):
    n = len(state["dtu_buffer"])
    if n < args.dtst_query_start:
        return False

    query = ((n - args.dtst_query_start) % args.dtst_query_step == 0)
    force = n > args.dtst_meta_max_update
    if not query and not force:
        return False

    old_scores = []
    new_scores = []
    was_training = student.training
    student.eval()
    with torch.no_grad():
        for x_cpu, old in state["dtu_buffer"]:
            x = x_cpu.to(device=device, non_blocking=True)
            new_scores.append(dtst_ent_proxy(student, x))
            old_scores.append(float(old))
    if was_training:
        student.train()

    old_arr = np.asarray(old_scores, dtype=np.float64)
    new_arr = np.asarray(new_scores, dtype=np.float64)
    improved_fraction = float((new_arr > old_arr).mean()) if len(old_arr) else 0.0


    if improved_fraction > 0.5 or force:
        dtst_weight_ema(
            history_teacher, student, alpha=args.dtst_ema_weight
        )
        state["dtu_buffer"].clear()
        return True

    return False


def _quantile_hist_threshold(hist, q):
    total = int(hist.sum())
    if total <= 0:
        return 1.0
    target = int(round(total * q))
    target = max(0, min(total - 1, target))
    cs = np.cumsum(hist)
    idx = int(np.searchsorted(cs, target + 1, side="left"))
    return idx / float(len(hist) - 1)


@torch.no_grad()
def dtst_initial_tcr(history_teacher, loader, device, args):
    history_teacher.eval()
    records = []
    bins = int(args.dtst_hist_bins)
    class_hist = np.zeros((NUM_CLASSES, bins), dtype=np.int64)

    for batch in tqdm(loader, desc="DT-ST initial TCR", leave=False):
        x = batch[0].to(device, non_blocking=True)
        name = batch[1][0]

        logits = history_teacher(x)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        pred_np = pred[0].cpu().numpy().astype(np.uint8)
        conf_np = conf[0].cpu().numpy().astype(np.float32)

        for c in range(NUM_CLASSES):
            vals = conf_np[pred_np == c]
            if vals.size:
                idx = np.minimum(
                    (vals * (bins - 1)).astype(np.int64), bins - 1
                )
                class_hist[c] += np.bincount(idx, minlength=bins)

        records.append((name, pred_np, conf_np.astype(np.float16)))

    thresholds = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        thresholds[c] = _quantile_hist_threshold(
            class_hist[c], args.dtst_topk_candidate
        )

    history = {}
    candidate = {}
    for name, pred, conf16 in records:
        conf = conf16.astype(np.float32)
        cand = pred.copy()
        for c in range(NUM_CLASSES):
            m = pred == c
            cand[m & (conf < thresholds[c])] = IGNORE
        history[name] = pred
        candidate[name] = cand

    print("[DT-ST TCR] initial class confidence thresholds:",
          [round(float(x), 4) for x in thresholds])
    return history, candidate, thresholds.tolist()


def _class_iou_current_vs_history(current, previous, num_classes):
    vals = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        a = current == c
        b = previous == c
        union = np.logical_or(a, b).sum()
        if union > 0:
            vals[c] = np.logical_and(a, b).sum() / float(union)
        else:
            vals[c] = 0.0
    return vals


@torch.no_grad()
def dtst_refresh_tcr(history_teacher, loader, device, previous_history, args):
    history_teacher.eval()
    rows = []

    for batch in tqdm(loader, desc="DT-ST refresh TCR", leave=False):
        x = batch[0].to(device, non_blocking=True)
        name = batch[1][0]

        logits = history_teacher(x)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        prev = previous_history.get(name)
        if prev is None:
            rel = np.zeros(NUM_CLASSES, dtype=np.float32)
        else:
            rel = _class_iou_current_vs_history(pred, prev, NUM_CLASSES)
        rows.append((name, pred, rel))

    rel_th = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        vals = np.asarray(
            [r[c] for _, _, r in rows if r[c] > 0],
            dtype=np.float32
        )
        if vals.size:
            vals.sort()
            idx = int(round(vals.size * args.dtst_topk_candidate))
            idx = max(0, min(vals.size - 1, idx))
            rel_th[c] = vals[idx]
        else:
            rel_th[c] = 1.0

    new_history = {}
    new_candidate = {}
    for name, pred, rel in rows:
        cand = pred.copy()
        for c in range(NUM_CLASSES):
            if rel[c] < rel_th[c]:
                cand[pred == c] = IGNORE
        new_history[name] = pred
        new_candidate[name] = cand

    print("[DT-ST TCR] refreshed ReL thresholds:",
          [round(float(x), 4) for x in rel_th])
    return new_history, new_candidate, rel_th.tolist()


def dtst_candidate_batch(names, candidate, shape_hw, device):
    arr = []
    H, W = shape_hw
    for name in names:
        x = candidate.get(name)
        if x is None:
            x = np.full((H, W), IGNORE, dtype=np.uint8)
        if x.shape != (H, W):

            xt = torch.from_numpy(x.astype(np.float32))[None, None]
            xt = F.interpolate(xt, size=(H, W), mode="nearest")[0, 0]
            x = xt.numpy().astype(np.uint8)
        arr.append(torch.from_numpy(x.astype(np.int64)))
    return torch.stack(arr, dim=0).to(device)


@torch.no_grad()
def dtst_classmix_from_candidates(x, cand, running_conf):
    B = x.shape[0]
    if B < 2:
        return x, torch.full_like(cand, IGNORE)

    perm = torch.randperm(B, device=x.device)
    mixed_x = x.clone()
    mix_label = torch.full_like(cand, IGNORE)

    for b in range(B):
        d = int(perm[b].item())
        donor = cand[d]
        classes = torch.unique(donor)
        classes = classes[classes != IGNORE]
        if classes.numel() == 0:
            continue


        probs = (1.0 - running_conf[classes.long()]).clamp_min(1e-8)
        probs = probs / probs.sum()
        n_choose = max(1, int(math.ceil(classes.numel() / 2.0)))
        n_choose = min(n_choose, classes.numel())
        idx = torch.multinomial(probs, n_choose, replacement=False)
        chosen = classes[idx]

        mask = torch.zeros_like(donor, dtype=torch.bool)
        for c in chosen:
            mask |= donor == c

        if mask.any():
            mixed_x[b] = torch.where(mask.unsqueeze(0), x[d], x[b])
            mix_label[b] = torch.where(mask, donor, mix_label[b])

    return mixed_x, mix_label


def dtst_step(student, history_teacher, x, names, optimizer, state, args, device):
    student.train()
    history_teacher.eval()


    with torch.no_grad():
        was_training = student.training
        student.eval()
        old_proxy = dtst_ent_proxy(student, x)
        if was_training:
            student.train()
        state["dtu_buffer"].append((x.detach().cpu(), old_proxy))

        tlogits = history_teacher(x)
        tlogits = F.interpolate(
            tlogits, size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        tprob = F.softmax(tlogits, dim=1)

        dtst_update_running_conf(tprob, state["running_conf"], args)
        pseudo = dtst_positive_pseudo(tprob, state["running_conf"], args)

        cand = dtst_candidate_batch(
            names, state["candidate"], x.shape[-2:], device
        )
        mixed_x, mix_label = dtst_classmix_from_candidates(
            x, cand, state["running_conf"]
        )


        pseudo_train = pseudo.clone()
        m = mix_label != IGNORE
        pseudo_train[m] = mix_label[m]

    slogits = student(mixed_x)
    slogits = F.interpolate(
        slogits, size=x.shape[-2:], mode="bilinear", align_corners=True
    )


    pos_map = F.cross_entropy(
        slogits, pseudo_train.long(), ignore_index=IGNORE, reduction="none"
    )
    loss_pos = pos_map.mean()


    loss_neg_raw = dtst_negative_bce(slogits, tprob.detach())
    loss = loss_pos + args.dtst_negative_weight * loss_neg_raw

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dtst_dynamic_teacher_update(history_teacher, student, state, args, device)

    return loss.item()


def build_models(checkpoint, device):
    student = build_full_model(pretrained_backbone=False)
    teacher = copy.deepcopy(student)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher_state = select_state(payload)
    student_state = payload["model_state"] if "teacher_state" in payload else teacher_state
    student.load_state_dict(clean_dlv2_keys(student_state))
    teacher.load_state_dict(clean_dlv2_keys(teacher_state))
    teacher.requires_grad_(False).eval()
    return student.to(device), teacher.to(device)


def train_pair_stage(opts, method, domain, incoming, stage_dir, device):
    set_seed(opts.seed)
    stage_dir.mkdir(parents=True)
    best_path = stage_dir / "best.pth"
    student, teacher = build_models(incoming, device)
    train_set = TargetImages(domain, opts.steps * opts.batch_size)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ValidationDataset(domain), batch_size=1,
                            num_workers=4, pin_memory=True)

    batches = iter(train_loader)

    if method == "csfda":
        optimizer = optim.SGD(configure_csfda_bn_only(student), lr=opts.csfda_lr,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        state = {"mu_e": opts.csfda_mu_e}
    else:
        student.requires_grad_(True)
        optimizer = optim.SGD([
            {"params": student.backbone.parameters(), "lr": BASE_LR},
            {"params": student.classifier.parameters(), "lr": HEAD_LR_MULT * BASE_LR},
        ], lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        tcr_loader = DataLoader(TargetImages(domain), batch_size=1,
                                num_workers=4, pin_memory=True)
        history, candidate, _ = dtst_initial_tcr(teacher, tcr_loader, device, opts)
        state = {
            "running_conf": torch.full((NUM_CLASSES,), opts.dtst_threshold_beta, device=device),
            "dtu_buffer": [],
            "history": history,
            "candidate": candidate,
        }

    best_miou = -1.0
    progress = tqdm(range(opts.steps), desc=f"{method} {domain}")

    for step in progress:
        if method == "dtst":
            adjust_learning_rate(optimizer, step, opts.steps)
            if step and step % opts.dtst_tcr_update_frequency == 0:
                state["history"], state["candidate"], _ = dtst_refresh_tcr(
                    teacher, tcr_loader, device, state["history"], opts
                )

        images, names = next(batches)
        images = images.to(device, non_blocking=True)

        if method == "csfda":
            loss = csfda_step(student, teacher, images, optimizer, state, opts)
        else:
            loss = dtst_step(student, teacher, images, names, optimizer, state, opts, device)
        progress.set_postfix(loss=f"{loss:.3f}")

        if (step > 0 and step % opts.eval_every == 0) or step == opts.steps - 1:
            score = evaluate(teacher, val_loader, device)["mIoU"]
            print(f"{method} {domain} step {step + 1}: {score:.2f}")
            if score > best_miou:
                best_miou = score
                save_ckpt(best_path, student, teacher, step + 1, score)

    return best_path


def split_model(model: nn.Module) -> Tuple[FeatureExtractor, Classifier]:
    return FeatureExtractor(copy.deepcopy(model.backbone)), Classifier(copy.deepcopy(model.classifier))


def freeze_bn(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            if m.affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)


def set_student_train(fea: nn.Module, cls: nn.Module) -> None:
    fea.train()
    cls.train()
    freeze_bn(fea)
    freeze_bn(cls)


def poly_lr(iteration: int, max_iter: int) -> float:
    return BASE_LR * max(1.0 - float(iteration) / float(max_iter), 0.0) ** POLY_POWER


class ConfidenceHistogram:
    def __init__(self, bins: int = CONF_HIST_BINS):
        self.bins = int(bins)
        self.counts = np.zeros((NUM_CLASSES, self.bins), dtype=np.uint64)
    def add(self, pred: np.ndarray, conf: np.ndarray):
        q = np.clip(np.rint(conf * (self.bins - 1)), 0, self.bins - 1).astype(np.int32)
        for c in range(NUM_CLASSES):
            vals = q[pred == c]
            if vals.size:
                self.counts[c] += np.bincount(vals, minlength=self.bins).astype(np.uint64)
    def quantile(self, cls: int, frac: float) -> float:
        hist = self.counts[cls]
        n = int(hist.sum())
        if n == 0:
            return 0.0

        rank = min(max(int(round(n * frac)), 0), n - 1)
        idx = int(np.searchsorted(np.cumsum(hist), rank + 1, side="left"))
        return float(idx) / float(self.bins - 1)


class CandidateStore:
    def __init__(self, stage_dir: Path, index: TargetIndex):
        self.root = stage_dir
        self.index = index
        self.ctr = stage_dir / "CTR"
        self.ctr_o = stage_dir / "CTR_O"
        self.meta = stage_dir / "meta_val"
        self.meta_mix = stage_dir / "meta_val_mixup"
        self.tmp_pred = stage_dir / "_candidate_tmp_pred"
        for d in (self.ctr, self.ctr_o, self.meta, self.meta_mix):
            d.mkdir(parents=True, exist_ok=True)
    def path(self, folder: Path, sid: str) -> Path:
        return folder / f"{sid}.png"
    def clear(self, *folders: Path):
        for folder in folders:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
    def save_mask(self, folder: Path, sid: str, arr: np.ndarray):
        Image.fromarray(arr.astype(np.uint8), mode="L").save(self.path(folder, sid))
    def load_mask(self, folder: Path, sid: str) -> np.ndarray:
        return np.asarray(Image.open(self.path(folder, sid)), dtype=np.uint8)
    def label_info(self, folder: Path) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
        l2f = {c: [] for c in range(NUM_CLASSES)}
        f2l: Dict[str, List[int]] = {}
        for sid in self.index.ids:
            p = self.path(folder, sid)
            if not p.is_file():
                continue
            arr = np.asarray(Image.open(p), dtype=np.uint8)
            classes = sorted(int(c) for c in np.unique(arr) if 0 <= int(c) < NUM_CLASSES)
            f2l[sid] = classes
            for c in classes:
                l2f[c].append(sid)
        return l2f, f2l
    def persist_info(self, folder: Path, name: str):
        with open(self.root / f"{name}.p", "wb") as f:
            pickle.dump(self.label_info(folder), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_info(self, name):
        with (self.root / f"{name}.p").open("rb") as stream:
            return pickle.load(stream)


def normalized_full_from_path(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((FULL_HW[1], FULL_HW[0]), Image.BILINEAR)
    return pil_to_normalized(im)


@torch.no_grad()
def predict_native(fea: nn.Module, cls: nn.Module, path: Path, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    im = Image.open(path).convert("RGB")
    native_w, native_h = im.size
    x = im.resize((FULL_HW[1], FULL_HW[0]), Image.BILINEAR)
    x = pil_to_normalized(x).unsqueeze(0).to(device, non_blocking=True)
    _, f = fea(x)
    logits = cls(f, (native_h, native_w))
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    return pred[0].cpu().numpy().astype(np.uint8), conf[0].cpu().numpy().astype(np.float32)


def _resize_label(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((hw[1], hw[0]), Image.NEAREST), dtype=np.uint8)


def _resize_float_nearest(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=hw, mode="nearest")[0, 0].numpy()


def _iou_per_class(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        p = pred == c
        t = target == c
        union = np.logical_or(p, t).sum()
        if union:
            out[c] = np.logical_and(p, t).sum() / float(union)
    return out


@torch.no_grad()
def generate_initial_candidates(store: CandidateStore, fea_his: nn.Module, cls_his: nn.Module, device: torch.device):
    hist = ConfidenceHistogram()
    fea_his.eval()
    cls_his.eval()
    for sid in tqdm(store.index.ids, desc="candidate:init-thresholds"):
        pred_native, conf_native = predict_native(fea_his, cls_his, store.index.id_to_path[sid], device)
        pred_small = _resize_label(pred_native, CANDIDATE_INIT_HW)
        conf_small = _resize_float_nearest(conf_native, CANDIDATE_INIT_HW)
        hist.add(pred_small, conf_small)
    thres = np.array([hist.quantile(c, TCR_TOPK_CANDIDATE) for c in range(NUM_CLASSES)], dtype=np.float64)
    meta_thres = np.array([hist.quantile(c, 0.9) for c in range(NUM_CLASSES)], dtype=np.float64)

    store.clear(store.ctr, store.ctr_o, store.meta, store.meta_mix)
    valid_counts: Dict[str, int] = {}
    for sid in tqdm(store.index.ids, desc="candidate:init-write"):
        path = store.index.id_to_path[sid]
        with Image.open(path) as im:
            native_w, native_h = im.size
        pred_native, conf_native = predict_native(fea_his, cls_his, path, device)
        pred_small = _resize_label(pred_native, CANDIDATE_INIT_HW)
        conf_small = _resize_float_nearest(conf_native, CANDIDATE_INIT_HW)

        full_hist = _resize_label(pred_small, (native_h, native_w))
        store.save_mask(store.ctr_o, sid, full_hist)

        masked = pred_small.copy()
        for c in range(NUM_CLASSES):
            masked[(masked == c) & (conf_small < thres[c])] = IGNORE
        masked_native = _resize_label(masked, (native_h, native_w))
        store.save_mask(store.ctr, sid, masked_native)
        valid_counts[sid] = int((masked_native != IGNORE).sum())

        meta_mask = pred_small.copy()
        for c in range(NUM_CLASSES):
            meta_mask[(meta_mask == c) & (conf_small < meta_thres[c])] = IGNORE
        if float((meta_mask == IGNORE).mean()) <= 0.98:
            store.save_mask(store.meta_mix, sid, _resize_label(meta_mask, (native_h, native_w)))

    topk = int(len(store.index.ids) * META_TOP_K)
    top_ids = sorted(store.index.ids, key=lambda s: valid_counts.get(s, 0), reverse=True)[:topk]
    for sid in top_ids:
        shutil.copy2(store.path(store.ctr, sid), store.path(store.meta, sid))
    for folder, name in ((store.ctr, "CTR"), (store.ctr_o, "CTR_O"), (store.meta, "meta_val"), (store.meta_mix, "meta_val_mixup")):
        store.persist_info(folder, name)


@torch.no_grad()
def refresh_candidates(store: CandidateStore, fea_his: nn.Module, cls_his: nn.Module, device: torch.device):
    store.clear(store.tmp_pred)
    rel_all = np.zeros((len(store.index.ids), NUM_CLASSES), dtype=np.float64)
    for n, sid in enumerate(tqdm(store.index.ids, desc="candidate:rel")):
        path = store.index.id_to_path[sid]
        pred_native, _ = predict_native(fea_his, cls_his, path, device)
        prev = store.load_mask(store.ctr_o, sid)
        if prev.shape != pred_native.shape:
            prev = _resize_label(prev, pred_native.shape)
        rel_all[n] = _iou_per_class(pred_native, prev)
        h, w = pred_native.shape
        half = _resize_label(pred_native, (max(1, h // 2), max(1, w // 2)))
        stored = _resize_label(half, (h, w))
        store.save_mask(store.tmp_pred, sid, stored)

    thres = np.zeros(NUM_CLASSES, dtype=np.float64)
    meta_thres = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        x = np.sort(rel_all[:, c][rel_all[:, c] > 0])
        if x.size:
            thres[c] = x[min(int(round(x.size * TCR_TOPK_CANDIDATE)), x.size - 1)]
            meta_thres[c] = x[min(int(round(x.size * 0.9)), x.size - 1)]

    store.clear(store.ctr, store.ctr_o, store.meta, store.meta_mix)
    valid_counts: Dict[str, int] = {}
    for n, sid in enumerate(tqdm(store.index.ids, desc="candidate:refresh-write")):
        pred = store.load_mask(store.tmp_pred, sid)
        store.save_mask(store.ctr_o, sid, pred)
        rel = rel_all[n]
        masked = pred.copy()
        for c in range(NUM_CLASSES):
            if rel[c] < thres[c]:
                masked[masked == c] = IGNORE
        store.save_mask(store.ctr, sid, masked)
        valid_counts[sid] = int((masked != IGNORE).sum())

        meta_mask = pred.copy()
        for c in range(NUM_CLASSES):
            if rel[c] < meta_thres[c]:
                meta_mask[meta_mask == c] = IGNORE
        if float((meta_mask == IGNORE).mean()) <= 0.98:
            store.save_mask(store.meta_mix, sid, meta_mask)

    topk = int(len(store.index.ids) * META_TOP_K)
    top_ids = sorted(store.index.ids, key=lambda s: valid_counts.get(s, 0), reverse=True)[:topk]
    for sid in top_ids:
        shutil.copy2(store.path(store.ctr, sid), store.path(store.meta, sid))
    for folder, name in ((store.ctr, "CTR"), (store.ctr_o, "CTR_O"), (store.meta, "meta_val"), (store.meta_mix, "meta_val_mixup")):
        store.persist_info(folder, name)
    shutil.rmtree(store.tmp_pred, ignore_errors=True)


def build_balanced_ids(label_to_file: Dict[int, List[str]], file_to_label: Dict[str, List[int]], max_iters: int) -> List[str]:

    sub_epoch = 500
    tmp: List[str] = []
    ind = {i: 0 for i in range(NUM_CLASSES)}
    pools = {c: list(label_to_file.get(c, [])) for c in range(NUM_CLASSES)}
    for _ in range(int(max_iters / sub_epoch) + 1):
        cur = np.zeros(NUM_CLASSES, dtype=np.float64)
        for _ in range(sub_epoch):
            dist = cur.copy() if cur.sum() == 0 else cur / cur.sum()
            w = 1.0 / np.log(1.0 + 1e-2 + dist)
            w /= w.sum()
            c = int(np.random.choice(NUM_CLASSES, p=w))
            files = pools[c]
            if len(files) <= 1:
                continue
            if ind[c] > len(files) - 1:
                np.random.shuffle(files)
                ind[c] = ind[c] % (len(files) - 1)
            sid = files[ind[c]]
            tmp.append(sid)
            ind[c] += 1
            for cc in file_to_label.get(sid, []):
                if 0 <= int(cc) < NUM_CLASSES:
                    cur[int(cc)] += 1
    if not tmp:
        raise RuntimeError("Stable-ST RSC produced no target samples; candidate masks are empty")
    return tmp


class StableMixer:
    def __init__(self, store: CandidateStore, folder: Path, meta: bool):
        self.store = store
        self.folder = folder
        self.meta = bool(meta)
        if folder == store.ctr:
            key = "CTR"
        elif folder == store.ctr_o:
            key = "CTR_O"
        elif folder == store.meta:
            key = "meta_val"
        elif folder == store.meta_mix:
            key = "meta_val_mixup"
        else:
            key = folder.name
        self.label_to_file, _ = store.load_info(key)
    def mix(self, in_img: np.ndarray, classes: Sequence[int], choice_p: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        import cv2
        from skimage.measure import label as sklabel, regionprops

        out_img = copy.deepcopy(np.asarray(in_img, dtype=np.uint8))
        out_lab = np.ones(out_img.shape[:2], dtype=np.float32) * IGNORE
        in_w, in_h = out_img.shape[:2]
        active = [int(c) for c in classes if len(self.label_to_file.get(int(c), [])) > 0]
        if not active:
            return out_img, out_lab
        p = np.asarray([choice_p[list(classes).index(c)] for c in active], dtype=np.float64)
        if not np.isfinite(p).all() or p.sum() <= 0:
            p = np.ones(len(active), dtype=np.float64)
        p /= p.sum()
        class_idx = int(np.random.choice(active, size=1, p=p)[0])

        cdd_img = cdd_lbl = mask = None
        for _ in range(2000):
            sid = random.choice(self.label_to_file[class_idx])
            cdd_img_pil = Image.open(self.store.index.id_to_path[sid]).convert("RGB")
            cdd_lbl_pil = Image.open(self.store.path(self.folder, sid))
            if cdd_lbl_pil.size != cdd_img_pil.size:
                cdd_lbl_pil = cdd_lbl_pil.resize(cdd_img_pil.size, Image.NEAREST)
            cdd_img = np.asarray(cdd_img_pil, dtype=np.uint8)
            cdd_lbl = np.asarray(cdd_lbl_pil, dtype=np.uint8)
            mask = cdd_lbl == class_idx
            if int(mask.sum()) >= 50:
                break
        else:
            return out_img, out_lab


        seg = sklabel(mask.astype(np.uint8), background=0, connectivity=2)
        regions = regionprops(seg)
        filled = np.zeros(out_img.shape[:2], dtype=np.uint8)
        boxes = []
        for region in regions:
            if int(region.area) < 20:
                continue
            y, x, y2, x2 = region.bbox
            h, w = y2 - y, x2 - x
            local_seg = seg[y:y2, x:x2]
            local_mask = (local_seg == region.label).astype(np.uint8)
            scale = 1.0 + np.random.rand() * (0.2 if self.meta else 1.5)
            crop_img = cv2.resize(cdd_img[y:y+h, x:x+w], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            crop_mask = cv2.resize(local_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            crop_lab = cv2.resize(cdd_lbl[y:y+h, x:x+w], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            new_w, new_h = crop_img.shape[:2]
            if in_w <= new_w or in_h <= new_h:
                continue
            cnt = 100
            x1 = y1 = 0
            while cnt > 1:
                inside_classes = crop_lab[crop_mask > 0]
                if class_idx not in inside_classes and boxes:
                    box = random.choice(boxes)
                    x1 = random.randint(box[0], box[1] - 1)
                    y1 = random.randint(box[2], box[3] - 1)
                    if x1 + new_w < in_w - 1 and y1 + new_h < in_h - 1:
                        break
                x1 = random.randint(0, in_w - new_w - 1)
                y1 = random.randint(0, in_h - new_h - 1)
                if filled[x1, y1] == 0 and filled[x1 + new_w, y1 + new_h] == 0:
                    break
                cnt -= 1
            if cnt > 1:
                region_i = out_img[x1:x1+new_w, y1:y1+new_h]
                region_l = out_lab[x1:x1+new_w, y1:y1+new_h]
                m = crop_mask > 0
                region_i[m] = crop_img[m]
                region_l[m] = crop_lab[m]
                filled[x1:x1+new_w, y1:y1+new_h] = 1
                vals = crop_lab[m]
                if len(np.unique(vals)) == 1 and class_idx in vals:
                    boxes.append([x1, x1 + new_w, y1, y1 + new_h])
        return out_img, out_lab


class StableTrainDataset(Dataset):
    def __init__(self, store: CandidateStore, ids: List[str]):
        self.store = store
        self.ids = ids
        self.mixer = StableMixer(store, store.ctr, meta=False)
        self.mix_classes = list(MIX_CLASSES_7)
        self.mix_p = np.ones(len(self.mix_classes), dtype=np.float64) / len(self.mix_classes)
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sid = self.ids[idx]
        image = Image.open(self.store.index.id_to_path[sid]).convert("RGB").resize((FULL_HW[1], FULL_HW[0]), Image.BILINEAR)
        img_full = pil_to_normalized(image)
        weak, geom = official_target_weak_view(copy.deepcopy(image))

        den = weak.clone()
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        den = ((den * std + mean) * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        mixed, mix_label = self.mixer.mix(den, self.mix_classes, self.mix_p)
        mixed = torch.from_numpy((mixed.astype(np.float32) / 255.0).transpose(2, 0, 1))
        mixed = (mixed - mean) / std
        return mixed.float(), sid, geom, img_full.float(), torch.from_numpy(mix_label).long()


def sample_meta_images(store: CandidateStore, meta_train_psd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, meta_f2l = store.load_info("meta_val")
    meta_mix_l2f, _ = store.load_info("meta_val_mixup")
    names = [sid for sid, labs in meta_f2l.items() if labs]
    if not names:
        raise RuntimeError("Stable-ST meta_val set is empty after candidate generation")
    meta_mixer = StableMixer(store, store.meta_mix, meta=True)

    unique_all = set(int(x) for x in torch.unique(meta_train_psd).detach().cpu().tolist() if 0 <= int(x) < NUM_CLASSES)
    probs = np.array([1.0 if c in unique_all else 0.0 for c in MIX_CLASSES_7], dtype=np.float64)

    if probs.sum() <= 0:
        probs[:] = 1.0
    probs /= probs.sum()

    out_x, out_y = [], []
    for b in range(meta_train_psd.shape[0]):
        cls_set = set(int(x) for x in torch.unique(meta_train_psd[b]).detach().cpu().tolist())
        match = np.array([len(cls_set & set(meta_f2l[sid])) for sid in names], dtype=np.float64)
        if match.max(initial=0) > 0:
            p = np.power(match / match.max(), 3)
            p = p / p.sum()
        else:
            p = np.ones(len(names), dtype=np.float64) / len(names)
        sid = str(np.random.choice(names, p=p))
        image = Image.open(store.index.id_to_path[sid]).convert("RGB").resize((FULL_HW[1], FULL_HW[0]), Image.BILINEAR)
        label = Image.open(store.path(store.meta, sid)).resize((FULL_HW[1], FULL_HW[0]), Image.NEAREST)
        arr_img = np.asarray(image, dtype=np.uint8)
        arr_lab = np.asarray(label, dtype=np.uint8)

        active = [c for c in MIX_CLASSES_7 if meta_mix_l2f.get(c)]
        if active:
            mix_img, mix_lab = meta_mixer.mix(arr_img, MIX_CLASSES_7, probs)
            arr_lab = arr_lab * (mix_lab == IGNORE) + mix_lab * (mix_lab != IGNORE)
            arr_img = mix_img
        x = torch.from_numpy((arr_img.astype(np.float32) / 255.0).transpose(2, 0, 1))
        x = (x - torch.tensor(MEAN).view(3,1,1)) / torch.tensor(STD).view(3,1,1)
        out_x.append(x.unsqueeze(0).float())
        out_y.append(torch.from_numpy(arr_lab.astype(np.int64)).unsqueeze(0))
    return torch.cat(out_x, 0), torch.cat(out_y, 0)


@torch.no_grad()
def update_running_conf(probs: torch.Tensor, running_conf: torch.Tensor) -> torch.Tensor:
    avg = probs.mean(0).view(NUM_CLASSES, -1).mean(-1)
    update = avg > 1e-8
    new = update & (running_conf == THRESHOLD_BETA)
    running_conf[new] = avg[new]
    running_conf.mul_(STAT_MOMENTUM).add_(avg, alpha=1.0 - STAT_MOMENTUM)
    return running_conf


def low_freq_amp(x: torch.Tensor, keep_ratio: float = NFD_STYLE_KEEP) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    xf = torch.fft.rfft2(gray, norm="ortho").abs()
    h, w = xf.shape[-2:]
    return xf[:, :, :max(1, int(h * keep_ratio)), :max(1, int(w * keep_ratio))].flatten(1)


@torch.no_grad()
def nfd_preupdate(fea: nn.Module, cls: nn.Module, full: torch.Tensor) -> Tuple[List[float], torch.Tensor, torch.Tensor]:

    _, f0 = fea(full)
    logits = cls(f0)
    probs = F.softmax(logits, dim=1)
    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(1).mean((1, 2))
    style = F.normalize(low_freq_amp(full), dim=1)
    b = full.shape[0]
    stable_pool = torch.argsort(ent)[:max(1, b // 2)]
    neigh_idx = []
    for i in range(b):
        sim = (style[i:i+1] @ style[stable_pool].T).squeeze(0)
        self_mask = stable_pool == i
        if self_mask.any():
            sim = sim.masked_fill(self_mask, -1.0)
        if torch.isneginf(sim).all() or (sim <= -1).all():
            choices = [j for j in range(b) if j != i]
            j = choices[0] if choices else i
        else:
            j = int(stable_pool[torch.argmax(sim)].item())
        neigh_idx.append(j)
    neigh = full[torch.tensor(neigh_idx, device=full.device)]
    _, fq = fea(full)
    _, fn = fea(neigh)
    s = (fq.permute(0,2,3,1) * fn.permute(0,2,3,1)).sum(-1)
    p = F.softmax(s.view(b, -1), dim=1)
    nfd = 1.0 - (-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).mean(1))
    return nfd.detach().cpu().tolist(), full.detach().cpu(), neigh.detach().cpu()


@torch.no_grad()
def nfd_measure(fea: nn.Module, query_cpu: torch.Tensor, neigh_cpu: torch.Tensor, device: torch.device) -> List[float]:
    q = query_cpu.to(device, non_blocking=True)
    n = neigh_cpu.to(device, non_blocking=True)
    _, fq = fea(q)
    _, fn = fea(n)
    s = (fq.permute(0,2,3,1) * fn.permute(0,2,3,1)).sum(-1)
    p = F.softmax(s.view(s.shape[0], -1), dim=1)
    val = 1.0 - (-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).mean(1))
    return val.detach().cpu().tolist()


@torch.no_grad()
def ema_step(dst: nn.Module, src: nn.Module, alpha: float):
    for p, sp in zip(dst.parameters(), src.parameters()):
        p.data.mul_(alpha)
        p.data.add_(sp.data, alpha=1.0-alpha)


class DTUState:
    def __init__(self):
        self.scores: List[float] = []
        self.eval_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.updates = 0
    def add(self, vals: List[float], query: torch.Tensor, neigh: torch.Tensor):
        self.scores.extend(float(x) for x in vals)
        self.eval_batches.append((query, neigh))
    @torch.no_grad()
    def maybe_update(self, fea: nn.Module, cls: nn.Module, fea_his: nn.Module, cls_his: nn.Module, device: torch.device) -> Tuple[bool, float]:
        n = len(self.scores)
        if n < DTU_QUERY_START or (n - DTU_QUERY_START) % DTU_QUERY_STEP != 0:
            return False, float("nan")
        now: List[float] = []
        was_train = fea.training
        fea.eval()
        for q, nb in self.eval_batches:
            now.extend(nfd_measure(fea, q, nb, device))
        if was_train:
            fea.train()
            freeze_bn(fea)
        old = np.asarray(self.scores, dtype=np.float64)
        cur = np.asarray(now, dtype=np.float64)
        frac = float(np.mean((cur - old) > 0)) if old.size else 0.0
        if frac > 0.5 or n > DTU_META_MAX_UPDATE:
            ema_step(cls_his, cls, DTU_EMA_WEIGHT)
            ema_step(fea_his, fea, DTU_EMA_WEIGHT)
            self.scores = []
            self.eval_batches = []
            self.updates += 1
            return True, frac
        return False, frac


@torch.no_grad()
def evaluate_split(fea: nn.Module, cls: nn.Module, domain: str, device: torch.device) -> dict:
    ds = ValidationDataset(domain)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    fea.eval()
    cls.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, y in tqdm(loader, desc=f"eval:{domain}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, f = fea(x)
        logits = cls(f)
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        pp = pred.cpu().numpy().astype(np.int64)
        yy = y.cpu().numpy().astype(np.int64)
        valid = (yy >= 0) & (yy < NUM_CLASSES)
        idx = NUM_CLASSES * yy[valid] + pp[valid]
        hist += np.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    diag = np.diag(hist).astype(np.float64)
    denom = hist.sum(1)+hist.sum(0)-diag
    iou = np.divide(diag, denom, out=np.full(NUM_CLASSES, np.nan), where=denom != 0)
    return {"mIoU": float(np.nanmean(iou) * 100.0),
            "class_iou": dict(zip(CLASS_NAMES, (iou * 100.0).tolist()))}


def metapl_pixel_weights(
    *, cls_meta, cls_his, train_feas, psd_label, mix_label, store, fea, criterion, device
):
    cls_meta.load_state_dict(cls_his.state_dict())
    cls_meta.train()
    params = OrderedDict((name, p.detach().requires_grad_(True))
                         for name, p in cls_meta.named_parameters())
    buffers = OrderedDict((name, b.detach()) for name, b in cls_meta.named_buffers())

    def predict(weights, features, shape):
        return functional_call(cls_meta, (weights, buffers), (features, shape))

    train_feas = train_feas.detach()
    shape = psd_label.shape[-2:]
    loss = criterion(predict(params, train_feas, shape), psd_label.long()).mean()
    grads = torch.autograd.grad(loss, tuple(params.values()))
    alpha = 0.1
    fast_params = OrderedDict(
        (name, (p - alpha * g).detach().requires_grad_(True))
        for (name, p), g in zip(params.items(), grads)
    )

    val_input, val_label = sample_meta_images(store, psd_label)
    val_input = val_input.to(device, non_blocking=True)
    val_label = val_label.to(device, non_blocking=True).long()
    with torch.no_grad():
        _, val_features = fea(val_input)
    val_pred = predict(fast_params, val_features, val_label.shape[-2:])
    val_loss = criterion(val_pred, val_label).mean()
    q = OrderedDict(zip(fast_params, torch.autograd.grad(val_loss, tuple(fast_params.values()))))

    def loss_map(weights):
        return criterion(predict(weights, train_feas, shape), psd_label.long())

    base = OrderedDict((name, p.detach()) for name, p in params.items())
    _, directional = jvp(loss_map, (base,), (q,), strict=False)
    grad = -(alpha / float(psd_label.numel())) * directional

    weights = torch.ones_like(grad)
    weights.addcdiv_(grad, grad.abs().add(1e-8), value=-0.9)
    if TCR_OPEN:
        weights[mix_label != IGNORE] = 1.0
    return weights.detach()


def build_stage_loader(store):
    label_to_file, file_to_label = store.load_info("CTR_O")
    ids = build_balanced_ids(label_to_file, file_to_label, OFFICIAL_LR_MAX_ITER * BATCH_SIZE)
    dataset = StableTrainDataset(store, ids)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    return dataset, loader


def train_stable_stage(opts, domain, incoming, stage_dir, device):
    set_seed(opts.seed + tuple(TARGETS.values()).index(domain))
    stage_dir.mkdir(parents=True)
    best_path = stage_dir / "best.pth"
    model = load_full_model(incoming, device)
    fea, cls = split_model(model)
    set_student_train(fea, cls)
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    fea_his, cls_his = split_model(teacher)
    cls_meta = copy.deepcopy(cls).requires_grad_(True)

    opt_f = optim.SGD((p for p in fea.parameters() if p.requires_grad),
                      lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    opt_c = optim.SGD((p for p in cls.parameters() if p.requires_grad),
                      lr=BASE_LR * HEAD_LR_MULT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")

    store = CandidateStore(stage_dir, make_target_index(domain))
    generate_initial_candidates(store, fea_his, cls_his, device)
    dataset, loader = build_stage_loader(store)
    batches = iter(loader)
    running_conf = torch.full((NUM_CLASSES,), THRESHOLD_BETA, device=device)
    dtu = DTUState()
    best_miou = -1.0

    progress = tqdm(range(opts.stable_stop_iter), desc=f"stable-st {domain}")
    for step in progress:
        try:
            weak, _, geom, full, mix_label = next(batches)
        except StopIteration:
            batches = iter(loader)
            weak, _, geom, full, mix_label = next(batches)
        weak = weak.to(device, non_blocking=True)
        full = full.to(device, non_blocking=True)
        geom = geom.to(device)
        mix_label = mix_label.to(device)

        lr = poly_lr(step, OFFICIAL_LR_MAX_ITER)
        opt_f.param_groups[0]["lr"] = lr
        opt_c.param_groups[0]["lr"] = lr * HEAD_LR_MULT
        opt_f.zero_grad(set_to_none=True)
        opt_c.zero_grad(set_to_none=True)
        set_student_train(fea, cls)
        _, features = fea(weak)
        logits = cls(features, weak.shape[-2:])

        with torch.no_grad():
            scores, query, neighbors = nfd_preupdate(fea, cls, full)
            dtu.add(scores, query, neighbors)
            _, history_features = fea_his(full)
            history_logits = cls_his(history_features, full.shape[-2:])
            weak_logits = full_logits_to_weak(history_logits, geom)
            running_conf = update_running_conf(F.softmax(history_logits, dim=1), running_conf)
            pseudo = F.softmax(weak_logits, dim=1).argmax(1).detach()
            train_features = full2weak_dynamic(history_features, geom)

        pseudo = torch.where(mix_label != IGNORE, mix_label.long(), pseudo.long())
        weights = metapl_pixel_weights(
            cls_meta=cls_meta, cls_his=cls_his, train_feas=train_features,
            psd_label=pseudo, mix_label=mix_label, store=store, fea=fea,
            criterion=criterion, device=device,
        )
        loss = (criterion(logits, pseudo) * (0.5 + 0.5 * torch.exp(weights - 1.0))).mean()
        loss.backward()
        opt_f.step()
        opt_c.step()

        dtu.maybe_update(fea, cls, fea_his, cls_his, device)
        probability = 1.0 - running_conf[MIX_CLASSES_7]
        probability /= probability.sum().clamp_min(1e-12)
        dataset.mix_p = probability.detach().cpu().numpy().astype(np.float64)
        progress.set_postfix(loss=f"{loss.item():.3f}")

        iteration = step + 1
        if iteration % opts.eval_every == 0 or iteration == opts.stable_stop_iter:
            score = evaluate_split(fea_his, cls_his, domain, device)["mIoU"]
            print(f"stable-st {domain} step {iteration}: {score:.2f}")
            if score > best_miou:
                best_miou = score
                save_ckpt(best_path, model, teacher, iteration, score)
            set_student_train(fea, cls)

        if iteration < opts.stable_stop_iter and (
            (TCR_OPEN and iteration % TCR_UPDATE_FREQUENCY == 0)
            or (METAPL_OPEN and iteration % METAPL_VAL_UPDATE == 0)
        ):
            refresh_candidates(store, fea_his, cls_his, device)
            dataset, loader = build_stage_loader(store)
            batches = iter(loader)

    return best_path


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=TARGETS, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=5001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stable-stop-iter", type=int, default=5000)
    parser.add_argument("--stable-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.set_defaults(
        csfda_percentile=55.0,
        csfda_mu_e=1e-3,
        csfda_beta=1e-4,
        csfda_eta=0.995,
        csfda_lr=1e-4,
        csfda_views=12,
        csfda_brightness=0.4,
        csfda_contrast=0.4,
        csfda_saturation=0.4,
        csfda_hue=0.1,
        csfda_noise_std=0.1,
        dtst_threshold_beta=0.001,
        dtst_stat_momentum=0.9,
        dtst_run_conf_upper=0.80,
        dtst_run_conf_lower=0.20,
        dtst_negative_weight=0.5,
        dtst_ema_weight=0.999,
        dtst_query_start=10,
        dtst_query_step=5,
        dtst_meta_max_update=50,
        dtst_topk_candidate=0.80,
        dtst_tcr_update_frequency=1000,
        dtst_hist_bins=10001,
    )
    return parser


def main():
    global DATA_ROOT, BATCH_SIZE

    opts = get_argparser().parse_args()
    DATA_ROOT = opts.data_root
    BATCH_SIZE = opts.stable_batch_size
    device = torch.device(f"cuda:{opts.gpu_id}")
    domain = TARGETS[opts.dataset]
    stable = opts.method == "stable-st"
    torch.backends.cudnn.benchmark = stable

    if opts.eval:
        model = load_full_model(opts.ckpt, device, pretrained_backbone=stable)
        if stable:
            fea, cls = split_model(model)
            scores = evaluate_split(fea, cls, domain, device)
        else:
            loader = DataLoader(ValidationDataset(domain), batch_size=1,
                                num_workers=4, pin_memory=True)
            scores = evaluate(model, loader, device)
        print(f"{opts.method} {opts.dataset} mIoU: {scores['mIoU']:.2f}")
        print("  ".join(f"{name}: {score:.2f}" for name, score in scores["class_iou"].items()))
        return

    stage_dir = opts.output_dir or Path("checkpoints/baselines") / opts.method / opts.dataset
    if stable:
        checkpoint = train_stable_stage(opts, domain, opts.ckpt, stage_dir, device)
    else:
        checkpoint = train_pair_stage(opts, opts.method, domain, opts.ckpt, stage_dir, device)
    print(f"Checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()
