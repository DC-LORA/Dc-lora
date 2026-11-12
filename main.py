from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from datasets import Cityscapes, GTA5
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
from utils import EMA

from peft import LoraConfig, get_peft_model, PeftModel


TARGET_LORA = ['conv1', 'conv3', "downsample.0"]

class SelfTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, pseudo_dir):
        self.orig_dataset = orig_dataset
        self.pseudo_dir = pseudo_dir
        self.length = len(orig_dataset)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        image, _ = self.orig_dataset[idx]
        img_path = self.orig_dataset.images[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        pseudo_path = os.path.join(self.pseudo_dir, f"pseudo_{img_id}.png")
        pseudo_label = Image.open(pseudo_path)
        pseudo_label = np.array(pseudo_label, dtype=np.uint8)
        pseudo_label = torch.from_numpy(pseudo_label).long()
        return image, pseudo_label

def generate_pseudo_labels(model, data_loader, device, save_dir, tau=0.9):
    model.to(device)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(data_loader, desc="Generating Pseudo Labels")):
            images = images.to(device, dtype=torch.float32)
            logits = model(images)
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)

            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            preds_np = preds.cpu().numpy().astype(np.uint8)
            conf_np = conf.cpu().numpy().astype(np.float32)

            for b in range(images.shape[0]):
                img_idx = idx * images.shape[0] + b
                img_path = data_loader.dataset.images[img_idx]
                img_id = os.path.splitext(os.path.basename(img_path))[0]

                raw_path = os.path.join(save_dir, f"pseudo_{img_id}.png")
                conf_path = os.path.join(save_dir, f"conf_{img_id}.npy")

                pred_new = preds_np[b]
                conf_new = conf_np[b]
                valid_new = conf_new >= tau
                merged_mask = np.full_like(pred_new, 255, dtype=np.uint8)
                merged_conf = np.zeros_like(conf_new, dtype=np.float32)

                if os.path.isfile(conf_path) and os.path.isfile(raw_path):
                    conf_old = np.load(conf_path).astype(np.float32)
                    mask_old = np.array(Image.open(raw_path), dtype=np.uint8)
                    valid_old = (mask_old != 255) & (conf_old >= tau)
                    use_old = valid_old & (~valid_new | (conf_old >= conf_new))
                    use_new = valid_new & (~valid_old | (conf_new > conf_old))
                    merged_mask[use_old] = mask_old[use_old]
                    merged_conf[use_old] = conf_old[use_old]
                    merged_mask[use_new] = pred_new[use_new]
                    merged_conf[use_new] = conf_new[use_new]
                else:
                    merged_mask[valid_new] = pred_new[valid_new]
                    merged_conf[valid_new] = conf_new[valid_new]

                Image.fromarray(merged_mask).save(raw_path)
                np.save(conf_path, merged_conf)

                color_mask = data_loader.dataset.decode_target(merged_mask)
                Image.fromarray(color_mask).save(os.path.join(save_dir, f"pseudo_{img_id}_color.png"))

    print(f"Pseudo-labels updated in {save_dir}")


import types
def patch_peft_forward_for_seg(m):
    if not isinstance(m, PeftModel):
        return m
    if getattr(m, "_seg_shim", False):
        return m
    def _seg_forward(self, *args, **kwargs):
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.pop("pixel_values", kwargs.pop("input_ids", None))
            if x is None:
                raise TypeError("Expected images tensor as first arg or in 'pixel_values'/'input_ids'.")
        return self.base_model(x)
    m.forward = types.MethodType(_seg_forward, m)
    m._seg_shim = True
    return m

def attach_lora(model, name, r=16, alpha=64, dropout=0.05, targets=TARGET_LORA):
    targets = targets or TARGET_LORA
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias='none',
                     task_type="FEATURE_EXTRACTION", target_modules=targets)

    if isinstance(model, PeftModel):
        if name not in model.peft_config:
            model.add_adapter(name, cfg)
        model.set_adapter(name)
        return patch_peft_forward_for_seg(model)

    m = get_peft_model(model, cfg)
    if name != "default":
        m.add_adapter(name, cfg)
        m.set_adapter(name)

        if "default" in m.peft_config:
            try:
                m.delete_adapter("default")
            except Exception:
                pass
    else:
        m.set_adapter("default")

    return patch_peft_forward_for_seg(m)


def list_adapters(model):
    return list(getattr(model, "peft_config", {}).keys()) if isinstance(model, PeftModel) else []


def extract_peft_adapters_from_state(sd):
    names = set()
    for k in sd.keys():
        if ".lora_A." in k:
            names.add(k.split(".lora_A.")[1].split(".")[0])
        elif ".lora_B." in k:
            names.add(k.split(".lora_B.")[1].split(".")[0])
    return sorted(names)

def is_dlv2(opts):
    return "deeplabv2" in opts.model

def peel_to_core(k: str) -> str:
    while not k.startswith(("backbone.", "classifier.")) and "." in k:
        k = k.split(".", 1)[1]
    return k


def eval_model(opts, model, ema):
    return ema.ema if (opts.self_training and ema is not None) else model

def clean_dlv2_keys(state):
    cleaned = {}
    for k, v in state.items():
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    return cleaned
def eval_model_merged_if_possible(opts, model, ema):
    m = ema.ema if (opts.self_training and ema is not None) else model
    if opts.use_lora and isinstance(m, PeftModel):
        import copy
        mc = copy.deepcopy(m)
        if hasattr(mc, "merge_and_unload"):
            merged = mc.merge_and_unload()
            if merged is not None:
                mc = merged
        mc.eval()
        return mc
    return m
def get_argparser():
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_root", type=str, default= '',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes', 'gta'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")


    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])


    parser.add_argument("--total_itrs", type=int, default=20000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float,  default=2.5e-4,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", type=int, nargs=2, metavar=("H","W"),default=[512, 1024],
       help="validation resize height and width, e.g. --val_size 512 1024")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=6,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, nargs=2, metavar=("H","W"),default=[512, 1024],
      help="train crop height and width, e.g. --crop_size 768 1280")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")


    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')


    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')


    parser.add_argument("--self_training", action='store_true', default=False,
                        help="Enable self-training with pseudo labels")

    parser.add_argument("--pseudo_label_dir", type=str, default="pseudo_labels",
                        help="Directory to save/load pseudo labels for self-training")
    parser.add_argument( "--use_lora",action="store_true",default=False, help="If set, the *student* model is wrapped with LoRA adapters")

    parser.add_argument("--active_adapter", type=str, default=None, help="name of the LoRA adapter to train/use (e.g., cs, idd)")

    parser.add_argument("--new_adapter_per_domain", action="store_true", help="add a fresh adapter if it doesn't exist")





    parser.add_argument("--pl_refresh_interval", type=int, default=500,
                        help="Regenerate pseudo labels every N iterations")
    parser.add_argument("--pl_tau", type=float, default=0.90,
                        help="Confidence threshold for pseudo labels during refresh")


    return parser

def get_dataset(opts):


    if opts.dataset == 'cityscapes':
        H1, W1 = opts.crop_size
        H2, W2 = opts.crop_val
        train_transform = et.ExtCompose([
            et.ExtResize((H1, W1)),
            et.ExtRandomCrop(((H1, W1)), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        cs_root = os.path.join(opts.data_root, 'cityscapes')
        train_dst = Cityscapes(root=cs_root, split='train', transform=train_transform)

        val_transform = et.ExtCompose([
            et.ExtResize((H2, W2)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        cs_root = os.path.join(opts.data_root, 'cityscapes')
        val_dst = Cityscapes(root=cs_root, split='val', transform=val_transform)

        return train_dst, val_dst

    elif opts.dataset == 'gta':

        H1, W1 = opts.crop_size
        H2, W2 = opts.crop_val
        train_transform = et.ExtCompose([
            et.ExtResize((H1, W1)),
            et.ExtRandomCrop(((H1, W1)), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        gta_root = os.path.join(opts.data_root, 'gta')
        train_dst = GTA5(root=gta_root, transform=train_transform)


        val_transform = et.ExtCompose([
            et.ExtResize((H2, W2)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        cs_root = os.path.join(opts.data_root, 'cityscapes')
        val_dst = Cityscapes(root=cs_root, split='val', transform=val_transform)

        return train_dst, val_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

    score = metrics.get_results()
    return score, ret_samples

def build_lora_model(base_model):
    from peft import LoraConfig, get_peft_model
    import types

    peft_config = LoraConfig(
        r=16, lora_alpha=64, lora_dropout=0.05, bias='none',
        task_type="FEATURE_EXTRACTION",
        target_modules=TARGET_LORA,
    )
    lora_model = get_peft_model(base_model, peft_config)


    def _seg_forward(self, *args, **kwargs):
        if len(args) > 0:
            x = args[0]
        else:

            x = kwargs.get("pixel_values", kwargs.get("input_ids", None))
            if x is None:
                raise TypeError("Expected images tensor as first arg or in 'pixel_values'/'input_ids'.")
        return self.base_model(x)

    lora_model.forward = types.MethodType(_seg_forward, lora_model)

    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass

    return lora_model


def build_standard_model(model_name, num_classes, output_stride):

    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    return model

def strip_checkpoint_prefix(k: str) -> str:
    for prefix in ("base_model.model.model.", "base_model.model.", "model.model.", "model."):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


def cpu_state_dict(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def merged_state_dict(module):
    import copy
    m_cpu = copy.deepcopy(module).cpu()
    if hasattr(m_cpu, "merge_and_unload"):
        merged = m_cpu.merge_and_unload()
        if merged is not None:
            m_cpu = merged
    return {strip_checkpoint_prefix(k): v.detach().cpu() for k, v in m_cpu.state_dict().items()}


def save_checkpoint(model, ema, opts, tag="latest"):
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{tag}_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth"

    payload = {
        "model_state_full": cpu_state_dict(model),
        "model_state_merged": merged_state_dict(model),
    }

    if ema is not None:
        payload["ema_state"] = cpu_state_dict(ema.ema)
        payload["ema_state_merged"] = merged_state_dict(ema.ema)

    torch.save(payload, save_path)
    print(f"Checkpoint saved to {save_path}")

def peel_prefix(k: str) -> str:
    while not (k.startswith("backbone.") or k.startswith("classifier.")):
        if "." not in k:
            break
        k = k.split(".", 1)[1]
    return k

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 7
    elif opts.dataset.lower() == 'gta':
        opts.num_classes = 7

    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    H1, W1 = opts.crop_size
    H2, W2 = opts.crop_val


    if opts.self_training:

        pseudo_transform = et.ExtCompose([
            et.ExtResize((H1, W1)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        cs_root = os.path.join(opts.data_root, 'cityscapes')
        pseudo_dst = Cityscapes(root=cs_root, split='train', transform=pseudo_transform)


        standard_model = build_standard_model(opts.model, opts.num_classes, opts.output_stride)

        if not (opts.ckpt and os.path.isfile(opts.ckpt)):
            raise FileNotFoundError("--ckpt is required for --self_training and must point to a valid file.")

        checkpoint = torch.load(opts.ckpt, map_location="cpu", weights_only=False)
        if opts.new_adapter_per_domain:
            merged = (
                checkpoint.get("ema_state_merged")
                or checkpoint.get("model_state_merged")
                or checkpoint["model_state"]
            )
            checkpoint = {"model_state_merged": merged}

        if "deeplabv2" in opts.model:

            state = checkpoint.get("model_state_merged") or checkpoint.get("model_state")
            if state is None:
                raise KeyError("Checkpoint missing 'model_state_merged' / 'model_state' for DeepLabV2.")
            standard_model.load_state_dict(clean_dlv2_keys(state), strict=True)

        else:

            if 'model_state_merged' in checkpoint:
                print("Loading standard model weights from 'model_state_merged'.")
                state = checkpoint['model_state_merged']
                if is_dlv2(opts):
                    standard_model.load_state_dict(clean_dlv2_keys(state), strict=True)

                else:
                    clean_state = {peel_prefix(k): v for k, v in state.items()}
                    standard_model.load_state_dict(clean_state, strict=True)

            elif "model_state" in checkpoint:
                standard_model.load_state_dict(checkpoint["model_state"], strict=True)
            else:
                raise KeyError("Checkpoint is missing both 'model_state_merged' and 'model_state'.")


        standard_model = standard_model.to(device)


        teacher_ema = EMA(standard_model, decay=0.999)


        if 'deeplabv2' in opts.model:
            if "ema_state_merged" in checkpoint:
                print("[PSD][DLv2] Using ema_state_merged for teacher.")
                t = clean_dlv2_keys(checkpoint["ema_state_merged"])
                teacher_ema.ema.load_state_dict(t, strict=True)

            elif "ema_state" in checkpoint:
                print("[PSD][DLv2] Merging ema_state (PEFT) for teacher.")
                import copy
                t_cpu = copy.deepcopy(teacher_ema.ema).cpu()
                lora_teacher = attach_lora(t_cpu, "temp", r=16, alpha=64, dropout=0.0, targets=TARGET_LORA)
                lora_teacher.load_state_dict(checkpoint["ema_state"], strict=True)
                if hasattr(lora_teacher, "merge_and_unload"):
                    lora_teacher.merge_and_unload()
                merged = clean_dlv2_keys(lora_teacher.state_dict())
                teacher_ema.ema.load_state_dict(merged, strict=True)


            elif "model_state_merged" in checkpoint:
                print("[PSD][DLv2] No EMA. Using model_state_merged as teacher.")
                t = clean_dlv2_keys(checkpoint["model_state_merged"])
                teacher_ema.ema.load_state_dict(t, strict=True)


            else:
                print("[PSD][DLv2] WARNING: No EMA or merged state found. Teacher is randomly initialized.")


        if 'deeplabv2' not in opts.model:
            if "ema_state_merged" in checkpoint:
                t = {peel_prefix(k): v for k, v in checkpoint["ema_state_merged"].items()}
                teacher_ema.ema.load_state_dict(t, strict=True)
                print("[PSD] Using ema_state_merged for teacher.")
            elif "ema_state" in checkpoint:
                import copy
                t_cpu = copy.deepcopy(teacher_ema.ema).cpu()
                lora_teacher = attach_lora(t_cpu, "temp", r=16, alpha=64, dropout=0.0, targets=TARGET_LORA)
                lora_teacher.load_state_dict(checkpoint["ema_state"], strict=False)
                if hasattr(lora_teacher, "merge_and_unload"):
                    lora_teacher.merge_and_unload()
                t = {peel_prefix(k): v for k, v in lora_teacher.state_dict().items()}
                teacher_ema.ema.load_state_dict(t, strict=True)
                print("[PSD] Merged ema_state for teacher.")

        teacher_ema.ema = teacher_ema.ema.to(device)


        if not (
                os.path.exists(opts.pseudo_label_dir) and
                any(f.startswith("pseudo_") and f.endswith(".png")
                for f in os.listdir(opts.pseudo_label_dir))
        ):
            print(f"[PSD] No existing pseudo labels found → generating into '{opts.pseudo_label_dir}'")
            pseudo_loader = data.DataLoader(pseudo_dst,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=4)
            generate_pseudo_labels(teacher_ema.ema,
                                        pseudo_loader,
                                        device,
                                        opts.pseudo_label_dir)
        else:
            print(f"[PSD] Found pseudo labels in '{opts.pseudo_label_dir}', skipping generation.")


        train_dst = Cityscapes(root=cs_root, split='train', transform=pseudo_transform)
        train_dst = SelfTrainingDataset(train_dst, opts.pseudo_label_dir)
        val_transform = et.ExtCompose([
            et.ExtResize((H2, W2)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        cs_root_val = os.path.join(opts.data_root, 'cityscapes')
        val_dst = Cityscapes(root=cs_root_val, split='val', transform=val_transform)

        backbone = build_standard_model(opts.model, opts.num_classes, opts.output_stride)
        model = backbone
    else:
        backbone = build_standard_model(opts.model,
                                        opts.num_classes,
                                        opts.output_stride)

        model = backbone
        if opts.use_lora:
            adp_name = opts.active_adapter or "default"
            if opts.new_adapter_per_domain or not isinstance(model, PeftModel):
                model = attach_lora(model, adp_name, r=16, alpha=64, dropout=0.05)
            else:

                if adp_name in list_adapters(model):
                    model.set_adapter(adp_name)
                else:
                    model = attach_lora(model, adp_name, r=16, alpha=64, dropout=0.05)

            if opts.use_lora and isinstance(model, PeftModel):
                adp = getattr(model, "active_adapter", None) or (opts.active_adapter or "default")

        train_transform = et.ExtCompose([
            et.ExtResize((H1, W1)),
            et.ExtRandomCrop(((H1, W1)), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtResize((H2, W2)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst, val_dst = get_dataset(opts)


    pin = torch.cuda.is_available()
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,
                                num_workers=2, drop_last=True, pin_memory=pin)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False,
                                num_workers=2, pin_memory=pin)

    print(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    model = model.to(device)

    ckpt = None
    if opts.ckpt and os.path.isfile(opts.ckpt):
        if opts.self_training:
            ckpt = checkpoint
        else:
            ckpt = torch.load(opts.ckpt, map_location="cpu", weights_only=False)

        if 'deeplabv2' in opts.model:
            state = ckpt.get('model_state_merged') or ckpt.get('model_state')
            assert state is not None, "DLv2 ckpt missing model_state_merged/model_state"
            print("[DLv2] Loading merged/plain state into base model.")
            model.load_state_dict(clean_dlv2_keys(state), strict=True)
            if opts.use_lora:
                adp = opts.active_adapter or "default"
                model = attach_lora(model, adp, r=16, alpha=64, dropout=0.05, targets=TARGET_LORA)
                model.set_adapter(adp)

                if ('model_state_full' in ckpt) and (not opts.new_adapter_per_domain):
                    missing, unexpected = model.load_state_dict(ckpt['model_state_full'], strict=False)
                    print(f"[DLv2][PEFT restore] missing={len(missing)} unexpected={len(unexpected)}")
                    for m in missing[:5]: print("  missing:", m)
                    for u in unexpected[:5]: print("  unexpected:", u)

        elif opts.use_lora and 'model_state_full' in ckpt:

            print("Restoring student from full LoRA checkpoint and reconstructing adapters.")
            full = ckpt['model_state_full']

            prev_adapters = extract_peft_adapters_from_state(full)
            print(f"   • adapters in ckpt: {prev_adapters or '[]'}")

            if prev_adapters:
                for name in prev_adapters:
                    model = attach_lora(model, name, r=16, alpha=64, dropout=0.05)
            else:
                model = attach_lora(model, opts.active_adapter or "default", r=16, alpha=64, dropout=0.05, targets=TARGET_LORA)

            if opts.new_adapter_per_domain and opts.active_adapter and opts.active_adapter not in prev_adapters:
                print(f"   • adding new adapter: {opts.active_adapter}")
                model = attach_lora(model, opts.active_adapter, r=16, alpha=64, dropout=0.05, targets=TARGET_LORA)

            model.load_state_dict(full, strict=False)
            if opts.active_adapter:
                model.set_adapter(opts.active_adapter)

        else:

            merged_key = 'model_state_merged' if 'model_state_merged' in ckpt else 'model_state'
            if merged_key in ckpt:
                print(f"Loading merged model ({merged_key}).")
                state = ckpt[merged_key]
                if is_dlv2(opts):
                    model.load_state_dict(clean_dlv2_keys(state), strict=True)
                else:
                    clean_state = {peel_prefix(k): v for k, v in state.items()}
                    model.load_state_dict(clean_state, strict=True)

                if opts.use_lora:
                    model = attach_lora(model, opts.active_adapter or "default", r=16, alpha=64, dropout=0.05, targets=TARGET_LORA)
            else:
                raise KeyError("Checkpoint missing 'model_state_full' / 'model_state_merged' / 'model_state'")

        print(f"Model restored from {opts.ckpt}")
    else:
        print("[!] Retrain / start fresh for this run.")


    if opts.use_lora:
        adp_name = opts.active_adapter or "default"


        if not isinstance(model, PeftModel) or adp_name not in list_adapters(model):
            model = attach_lora(model, adp_name, r=16, alpha=64, dropout=0.05)
        else:
            model.set_adapter(adp_name)
    ema = None
    if opts.self_training:
        ema = EMA(model, decay=0.999)
        ema.ema.to(device)

        if ckpt is not None and ('ema_state' in ckpt) and (not opts.new_adapter_per_domain):
            try:
                ema.ema.load_state_dict(ckpt['ema_state'], strict=False)
                print("EMA restored from checkpoint.")
            except Exception as e:
                print(f"[Skip EMA restore] {e}")
        elif ckpt is not None and ('ema_state_merged' in ckpt) and not isinstance(model, PeftModel):
            try:
                ema.ema.load_state_dict(ckpt['ema_state_merged'], strict=True)
                print("EMA (merged) restored.")
            except Exception as e:
                print(f"[Skip EMA merged restore] {e}")


    backbone_ref = model.backbone
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(backbone_ref, momentum=0.01)


    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            m.momentum = 0.01


    metrics = StreamSegMetrics(opts.num_classes)


    try:
        from peft.utils import is_lora_parameter as _is_lora_name
        def is_lora_name(n: str) -> bool: return _is_lora_name(n)
    except Exception:
        def is_lora_name(n: str) -> bool: return "lora_" in n

    def is_active_lora_name(n: str, active: str) -> bool:

        return is_lora_name(n) and (f".{active}." in n)


    if opts.use_lora:

        for _, p in model.named_parameters():
            p.requires_grad = False

        active = getattr(model, "active_adapter", None)
        assert active is not None, "No active LoRA adapter set."


        lora_params = []
        for n, p in model.named_parameters():
            if is_active_lora_name(n, active):
                p.requires_grad = True
                lora_params.append(p)


        bn_affine = []
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.affine:
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)

                bn_affine += [m.weight, m.bias]

        bb_params, head_params = [], []
        param_groups = [
            {"params": lora_params, "lr": opts.lr,           "weight_decay": 0.0},
            {"params": bn_affine,   "lr": 0.1 * opts.lr,     "weight_decay": 0.0},
        ]
        trainable_params = lora_params + bn_affine
        mode_str = f"LoRA+BN-affine (active='{active}')"


    else:


        for _, p in model.named_parameters():
            p.requires_grad = True


        param_groups = [
            {"params": model.parameters(), "lr": opts.lr, "weight_decay": 5e-4},
        ]

        trainable_params = list(model.parameters())
        mode_str = "ST (Full Fine-Tune)"
        bb_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
        head_params = list(model.classifier.parameters()) if hasattr(model, "classifier") else []
        lora_params = []
        bn_affine = []


    optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    print(f"[MODE] {mode_str} | "
      f"backbone={sum(p.numel() for p in bb_params)} | "
      f"head={sum(p.numel() for p in head_params)} | "
      f"lora={sum(p.numel() for p in lora_params)} | "
      f"bn_affine={sum(p.numel() for p in bn_affine) if 'bn_affine' in locals() else 0}")


    assert any(p.requires_grad for p in model.parameters()), "No trainable parameters found!"

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=0.1
        )


    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    utils.mkdir('checkpoints')
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples).tolist() if opts.enable_vis else None


    interval_loss = 0.0
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for images, labels in train_loader:
            cur_itrs += 1


            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)

            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            if opts.self_training and ema is not None:
                ema.update(model)


            np_loss = loss.item()
            interval_loss += np_loss
            if vis: vis.vis_scalar('Loss', cur_itrs, np_loss)

            if cur_itrs % opts.print_interval == 0:
                print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{int(opts.total_itrs)}, Loss={interval_loss/opts.print_interval:.6f}")
                interval_loss = 0.0

            if cur_itrs % opts.val_interval == 0:

                save_checkpoint(model, ema, opts, tag="latest")
                print("validation...")
                m_eval = eval_model_merged_if_possible(opts, model, ema)
                m_eval.eval()

                val_score, ret_samples = validate(opts, m_eval, val_loader, device, metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score["Mean IoU"] > best_score:
                    best_score = val_score["Mean IoU"]
                    save_checkpoint(model, ema, opts, tag="best")

                if opts.self_training and ema is not None:
                    ema.ema.to(device)

            if opts.self_training and (cur_itrs % opts.pl_refresh_interval == 0):
                base_ds = train_dst.orig_dataset if isinstance(train_dst, SelfTrainingDataset) else train_dst
                base_loader = data.DataLoader(base_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=pin)
                teacher = ema.ema if ema is not None else model
                teacher.eval()
                generate_pseudo_labels(teacher, base_loader, device, opts.pseudo_label_dir, tau=opts.pl_tau)
                model.train()


            scheduler.step()
            if cur_itrs >= opts.total_itrs:
                save_checkpoint(model, ema, opts, tag="latest")
                return

if __name__ == '__main__':
    main()
