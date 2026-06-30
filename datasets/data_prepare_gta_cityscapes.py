import os
import glob
import numpy as np
from PIL import Image
# Shared 7-superclass IDs:
# 0=flat, 1=construction, 2=object, 3=nature,
# 4=sky, 5=human, 6=vehicle, 7=ignored

my_palette = np.zeros((256 * 3,), dtype=np.uint8)
my_palette[0*3:0*3+3] = (128, 64, 128)    # flat
my_palette[1*3:1*3+3] = (70, 70, 70)      # construction
my_palette[2*3:2*3+3] = (232, 174, 0)     # object
my_palette[3*3:3*3+3] = (107, 142, 35)    # nature
my_palette[4*3:4*3+3] = (70, 130, 180)    # sky
my_palette[5*3:5*3+3] = (220, 20, 60)     # human
my_palette[6*3:6*3+3] = (0, 0, 142)       # vehicle
my_palette[7*3:7*3+3] = (0, 0, 0)         # ignored

# Cityscapes mapping using SOTA column
# Original Cityscapes labelIds -> 7 superclass


cityscapes_id2superclass = {
    0: 7,   # unlabeled -> ignored
    1: 7,   # ego vehicle -> ignored
    2: 7,   # rectification border -> ignored
    3: 7,   # out of roi -> ignored
    4: 7,   # static -> ignored
    5: 7,   # dynamic -> ignored
    6: 7,   # ground -> ignored

    7: 0,   # road -> flat
    8: 0,   # sidewalk -> flat
    9: 7,   # parking -> ignored
    10: 7,  # rail track -> ignored

    11: 1,  # building -> construction
    12: 1,  # wall -> construction
    13: 1,  # fence -> construction
    14: 7,  # guard rail -> ignored
    15: 7,  # bridge -> ignored
    16: 7,  # tunnel -> ignored

    17: 2,  # pole -> object
    18: 7,  # polegroup -> ignored
    19: 2,  # traffic light -> object
    20: 2,  # traffic sign -> object

    21: 3,  # vegetation -> nature
    22: 3,  # terrain -> nature
    23: 4,  # sky -> sky

    24: 5,  # person -> human
    25: 5,  # rider -> human

    26: 6,  # car -> vehicle
    27: 6,  # truck -> vehicle
    28: 6,  # bus -> vehicle
    29: 7,  # caravan -> ignored
    30: 7,  # trailer -> ignored
    31: 6,  # train -> vehicle
    32: 6,  # motorcycle -> vehicle
    33: 6,  # bicycle -> vehicle

    255: 7  # ignore / void
}

# GTA mapping using SOTA column
# Original GTA IDs -> 7 superclass


gta_id2superclass = {
    0: 0,   # road -> flat
    1: 0,   # sidewalk -> flat

    2: 1,   # building -> construction
    3: 1,   # wall -> construction
    4: 1,   # fence -> construction

    5: 2,   # pole -> object
    6: 2,   # traffic light -> object
    7: 2,   # traffic sign -> object

    8: 3,   # vegetation -> nature
    9: 3,   # terrain -> nature
    10: 4,  # sky -> sky

    11: 5,  # person -> human
    12: 5,  # rider -> human

    13: 6,  # car -> vehicle
    14: 6,  # truck -> vehicle
    15: 6,  # bus -> vehicle
    16: 6,  # train -> vehicle
    17: 6,  # motorcycle -> vehicle
    18: 6,  # bicycle -> vehicle

    255: 7  # ignore / unlabeled if stored as 255
}


def remap_folder(input_dir, output_dir, mapping, pattern="**/*.png"):
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
    print(f"Found {len(image_files)} label files.")
    lut = np.full((256,), 7, dtype=np.uint8)

    for k, v in mapping.items():
        if 0 <= k <= 255:
            lut[k] = v

    for file_path in image_files:
        rel_path = os.path.relpath(file_path, input_dir)

        out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(out_subdir, exist_ok=True)

        file_name = os.path.basename(file_path)
        name_part, ext = os.path.splitext(file_name)

        out_id_path = os.path.join(out_subdir, file_name)
        out_color_path = os.path.join(out_subdir, name_part + "_7classColor" + ext)

        orig_label = Image.open(file_path)
        orig_array = np.array(orig_label)
        safe_array = np.where((orig_array >= 0) & (orig_array <= 255), orig_array, 255).astype(np.uint8)

        new_label = lut[safe_array]
        Image.fromarray(new_label).save(out_id_path)
        color_img = Image.fromarray(new_label, mode="P")
        color_img.putpalette(my_palette)
        color_img.save(out_color_path)

        print(f"Processed: {rel_path}")



# Run Cityscapes


cityscapes_input_dir = r""
cityscapes_output_dir = r""

remap_folder(
    input_dir=cityscapes_input_dir,
    output_dir=cityscapes_output_dir,
    mapping=cityscapes_id2superclass,
    pattern="**/*_gtFine_labelIds.png"
)



# Run GTA

gta_input_dir = r""
gta_output_dir = r""

remap_folder(
    input_dir=gta_input_dir,
    output_dir=gta_output_dir,
    mapping=gta_id2superclass,
    pattern="**/*.png"
)