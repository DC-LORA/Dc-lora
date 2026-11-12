from __future__ import annotations

import argparse
import os
from typing import List

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import network
import utils
from datasets import Cityscapes
from metrics import StreamSegMetrics


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="path to an image or a directory of images")
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        choices=["voc", "cityscapes"])
    parser.add_argument("--model", type=str, default="deeplabv3plus_resnet101",
                        choices=sorted(n for n in network.modeling.__dict__
                                       if n.islower() and not n.startswith("_")
                                       and callable(network.modeling.__dict__[n])))
    parser.add_argument("--separable_conv", action="store_true", default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--save_val_results_to", type=str, default=None)
    parser.add_argument("--crop_val", type=int, nargs=2, metavar=("H", "W"), default=[512, 1024])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser


def clean_dlv2_keys(state):
    cleaned = {}
    for k, v in state.items():
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    return cleaned


def peel_prefix(k: str) -> str:
    while not (k.startswith("backbone.") or k.startswith("classifier.")):
        if "." not in k:
            break
        k = k.split(".", 1)[1]
    return k


def has_lora_keys(state) -> bool:
    return isinstance(state, dict) and any(".lora_A." in k or ".lora_B." in k for k in state.keys())


def is_plain_state_dict(obj) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def select_inference_state(ckpt):
    if is_plain_state_dict(ckpt):
        if has_lora_keys(ckpt):
            raise KeyError("This checkpoint contains LoRA adapter weights only. Use a checkpoint with merged weights for inference.")
        return ckpt

    for key in ("ema_state_merged", "model_state_merged", "model_state", "state_dict"):
        if key in ckpt and ckpt[key] is not None:
            state = ckpt[key]
            if has_lora_keys(state):
                raise KeyError(f"'{key}' contains LoRA adapter weights. Use merged weights for inference.")
            return state

    if "model_state_full" in ckpt:
        state = ckpt["model_state_full"]
        if has_lora_keys(state):
            raise KeyError(
                "The checkpoint has LoRA-visible weights but no merged inference state. "
                "Please provide a checkpoint containing 'model_state_merged' or 'ema_state_merged'."
            )
        return state

    raise KeyError("No usable inference weights found in the checkpoint.")


def collect_images(path) -> List[str]:
    files = []
    if os.path.isdir(path):
        for root, _, fns in os.walk(path):
            for fn in fns:
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    files.append(os.path.join(root, fn))
    elif os.path.isfile(path):
        files = [path]

    uniq, seen = [], set()
    for f in files:
        key = os.path.abspath(f).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return sorted(uniq)


def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == "voc":
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    else:
        opts.num_classes = 7
        decode_fn = Cityscapes.decode_target

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_files = collect_images(opts.input)
    if not image_files:
        raise FileNotFoundError(f"No images found under: {opts.input}")
    print(f"Found {len(image_files)} images.")

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride
    )

    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)

    utils.set_bn_momentum(model.backbone, momentum=0.01)

    assert os.path.isfile(opts.ckpt), f"Checkpoint not found: {opts.ckpt}"
    ckpt = torch.load(opts.ckpt, map_location="cpu", weights_only=False)
    state = select_inference_state(ckpt)

    if "deeplabv2" in opts.model:
        model.load_state_dict(clean_dlv2_keys(state), strict=True)
    else:
        cleaned = {peel_prefix(k): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=True)

    print("[Inference] Loaded merged/plain checkpoint weights.")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    height, width = opts.crop_val
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if opts.save_val_results_to:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    metrics = StreamSegMetrics(opts.num_classes)
    metrics.reset()

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            logits = model(x)
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)

            pred = logits.argmax(1).cpu().numpy()[0]
            color = decode_fn(pred).astype("uint8")

            if opts.save_val_results_to:
                out_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                Image.fromarray(color).save(os.path.join(opts.save_val_results_to, out_name))

            if opts.dataset.lower() == "cityscapes":
                gt_path = img_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("leftImg8bit", "gtFine")
                if os.path.isfile(gt_path):
                    gt = Image.open(gt_path).resize((width, height), resample=Image.NEAREST)
                    gt_arr = Cityscapes.encode_target(gt)
                    metrics.update([gt_arr], [pred])

    if opts.dataset.lower() == "cityscapes":
        res = metrics.get_results()
        if res["Mean IoU"] > 0:
            print("\n=== Evaluation (Cityscapes 7-class) ===")
            print(f"Overall Acc       : {res['Overall Acc']:.4f}")
            print(f"Mean IoU          : {res['Mean IoU']:.4f}")

            class_iou = res["Class IoU"]
            for i, name in enumerate(["flat", "construction", "object", "nature", "sky", "human", "vehicle"]):
                print(f"  {name:12s}: IoU={class_iou[i]:.4f}")


if __name__ == "__main__":
    main()
