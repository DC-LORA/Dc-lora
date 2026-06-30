import os
import glob
import numpy as np
from PIL import Image

# 1. Mapping from original IDD IDs (0..39) to the 7 super-classes:
#    0=flat, 1=construction, 2=object, 3=nature, 4=sky, 5=human, 6=vehicle, 7=ignored
origId2superClass = {
    0: 0,   # road -> flat
    1: 7,   # parking -> ignored (adjust if needed)
    2: 7,   # drivable fallback -> flat
    3: 0,   # sidewalk -> flat
    4: 7,   # rail track -> ignored
    5: 7,   # non-drivable fallback -> ignored
    6: 5,   # person -> human
    7: 7,   # animal -> ignored
    8: 5,   # rider -> human
    9: 6,   # motorcycle -> vehicle
    10: 6,  # bicycle -> vehicle
    11: 7,  # autorickshaw -> vehicle
    12: 6,  # car -> vehicle
    13: 6,  # truck -> vehicle
    14: 6,  # bus -> vehicle
    15: 7,  # caravan -> vehicle
    16: 7,  # trailer -> vehicle
    17: 6,  #train - vehicle
    18: 7,  # vehicle fallback -> vehicle
    19: 7,  # curb -> construction
    20: 1,  # wall -> construction
    21: 1,  # fence -> construction
    22: 7,  # guard rail -> construction
    23: 7,  # billboard -> object
    24: 2,  # traffic sign -> object
    25: 2,  # traffic light -> object
    26: 2,  # pole -> object
    27: 7,  # polegroup -> object
    28: 7,  # obs-str-bar-fallback -> object
    29: 1,  # building -> construction
    30: 7,  # bridge -> construction
    31: 7,  # tunnel -> construction
    32: 3,  # vegetation -> nature
    33: 4,  # sky -> sky
    34: 7,  # fallback background -> ignored
    35: 7,  # unlabeled -> ignored
    36: 7,  # ego vehicle -> ignored
    37: 7,  # rectification border -> ignored
    38: 7,  # out of roi -> ignored
    39: 7   # license plate -> ignored
}

# 2. Define the color palette for the 7 super-classes (plus "ignored"):
#    Index: 0=flat, 1=construction, 2=object, 3=nature, 4=sky, 5=human, 6=vehicle, 7=ignored
my_palette = np.zeros((256 * 3,), dtype=np.uint8)
my_palette[0*3:0*3+3] = (128, 64, 128)    # flat
my_palette[1*3:1*3+3] = (70, 70, 70)      # construction
my_palette[2*3:2*3+3] = (153, 153, 153)   # object
my_palette[3*3:3*3+3] = (107, 142, 35)    # nature
my_palette[4*3:4*3+3] = (70, 130, 180)    # sky
my_palette[5*3:5*3+3] = (220, 20, 60)     # human
my_palette[6*3:6*3+3] = (0, 0, 142)       # vehicle
my_palette[7*3:7*3+3] = (0, 0, 0)         # ignored (black)


input_dir = r""
output_dir = r""
os.makedirs(output_dir, exist_ok=True)
pattern = os.path.join(input_dir, '**', '*_gtFine_labelIds.png')
image_files = glob.glob(pattern, recursive=True)

max_id = max(origId2superClass.keys()) 
lut = np.full((max_id + 1,), 7, dtype=np.uint8)  
for k, v in origId2superClass.items():
    lut[k] = v


for file_path in image_files:

    rel_path = os.path.relpath(file_path, input_dir)
    
    out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
    os.makedirs(out_subdir, exist_ok=True)

    file_name = os.path.basename(file_path)
    
    orig_label = Image.open(file_path)
    orig_array = np.array(orig_label, dtype=np.int32)

    orig_array = np.where((orig_array < 0) | (orig_array > max_id), max_id, orig_array)
  
    new_label = lut[orig_array]

    out_id_name = file_name
    name_part, ext = os.path.splitext(file_name)
    out_color_name = name_part + "_7classColor" + ext
    out_id_path = os.path.join(out_subdir, out_id_name)
    out_color_path = os.path.join(out_subdir, out_color_name)
    Image.fromarray(new_label).save(out_id_path)
    color_img = Image.fromarray(new_label, mode='P')
    color_img.putpalette(my_palette)
    color_img.save(out_color_path)

    print(f"Processed: {rel_path}")
