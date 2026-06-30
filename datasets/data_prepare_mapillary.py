import os
import glob
import json
import numpy as np
from PIL import Image


with open("", "r") as f:
    config = json.load(f)

color_to_id = {}
for idx, label in enumerate(config["labels"]):
    r, g, b = label["color"]
    key = (r << 16) + (g << 8) + b
    color_to_id[key] = idx

# Original mapping from original label IDs to 7 classes.
origId2superClass = {
    0: 7,   # bird -> other
    1: 7,   # ground animal -> other
    2: 7,   # curb -> construction
    3: 1,   # fence -> construction
    4: 7,   # guard rail -> construction
    5: 7,   # other barrier -> construction
    6: 1,   # wall -> construction
    7: 7,   # bike lane -> flat
    8: 7,   # crosswalk - plain -> flat
    9: 7,   # curb cut -> flat
    10: 7,  # parking -> flat
    11: 7,  # pedestrian area -> flat
    12: 7,  # rail track -> flat
    13: 0,  # road -> flat
    14: 7,  # service lane -> flat
    15: 0,  # sidewalk -> flat
    16: 7,  # bridge -> construction
    17: 1,  # building -> construction
    18: 7,  # tunnel -> construction
    19: 5,  # person -> human
    20: 5,  # bicyclist -> human
    21: 5,  # motorcyclist -> human
    22: 5,  # other rider -> human
    23: 0,  # lane marking - crosswalk zebra -> flat
    24: 0,  # lane marking - general -> other
    25: 7,  # mountain -> nature
    26: 7,  # sand -> nature
    27: 4,  # sky -> sky
    28: 7,  # snow -> nature
    29: 3,  # terrain -> nature
    30: 3,  # vegetation -> nature
    31: 7,  # water -> nature
    32: 7,  # banner -> object
    33: 7,  # bench -> object
    34: 7,  # bike rack -> object
    35: 7,  # billboard -> object
    36: 7,  # catch basin -> object
    37: 7,  # cctv camera -> object
    38: 7,  # fire hydrant -> object
    39: 7,  # junction box -> object
    40: 7,  # mailbox -> object
    41: 0,  # manhole -> object
    42: 7,  # phone booth -> object
    43: 7,  # pothole -> object
    44: 7,  # street light -> object
    45: 2,  # pole -> object
    46: 7,  # traffic sign frame -> object
    47: 2,  # utility pole -> object
    48: 2,  # traffic light -> object
    49: 7,  # traffic sign (back) -> object
    50: 2,  # traffic sign (front) -> object
    51: 7,  # trash can -> object
    52: 6,  # bicycle -> vehicle
    53: 7,  # boat -> vehicle
    54: 6,  # bus -> vehicle
    55: 6,  # car -> vehicle
    56: 7,  # caravan -> vehicle
    57: 6,  # motorcycle -> vehicle
    58: 6,  # on rails -> vehicle
    59: 7,  # other vehicle -> vehicle
    60: 7,  # trailer -> vehicle
    61: 6,  # truck -> vehicle
    62: 7,  # wheeled slow -> vehicle
    63: 7,  # car mount -> other
    64: 7,  # ego vehicle -> other
    -1: 7   # unlabeled -> ignored
}

my_palette = np.zeros((256 * 3,), dtype=np.uint8)
my_palette[0*3:0*3+3] = (128, 64, 128)    # 0: flat
my_palette[1*3:1*3+3] = (70, 70, 70)      # 1: construction
my_palette[2*3:2*3+3] = (153, 153, 153)   # 2: object
my_palette[3*3:3*3+3] = (107, 142, 35)    # 3: nature
my_palette[4*3:4*3+3] = (70, 130, 180)    # 4: sky
my_palette[5*3:5*3+3] = (220, 20, 60)     # 5: human
my_palette[6*3:6*3+3] = (0, 0, 142)       # 6: vehicle
my_palette[7*3:7*3+3] = (0, 0, 0)         # 7: ignored

input_dir = r""
output_dir = r""
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = glob.glob(os.path.join(input_dir, "*.png"))

for file_path in image_files:
    orig_img = Image.open(file_path).convert('RGB')
    orig_array = np.array(orig_img)
    
    img_idx = (orig_array[:, :, 0].astype(np.uint32) << 16) + \
              (orig_array[:, :, 1].astype(np.uint32) << 8) + \
              orig_array[:, :, 2].astype(np.uint32)

    label_ids = np.full(img_idx.shape, -1, dtype=np.int32)
    unique_vals = np.unique(img_idx)
    for val in unique_vals:
        if val in color_to_id:
            label_ids[img_idx == val] = color_to_id[val]
    min_id = -1
    max_defined = max(origId2superClass.keys())
    max_label = max(max_defined, int(label_ids.max()))
    lut_mapping = np.full((max_label - min_id + 1,), 7, dtype=np.uint8)  
    for orig_id, new_id in origId2superClass.items():
        lut_mapping[orig_id - min_id] = new_id
    new_label = lut_mapping[label_ids - min_id]
    file_name = os.path.basename(file_path)
    out_name_id = file_name.replace("_labelIds", "_labelIds")
    out_path_id = os.path.join(output_dir, out_name_id)
    Image.fromarray(new_label).save(out_path_id)
    out_name_color = file_name.replace("_labelIds", "_7class_color")
    out_path_color = os.path.join(output_dir, out_name_color)
    color_img = Image.fromarray(new_label, mode='P')
    color_img.putpalette(my_palette)
    color_img.save(out_path_color)
    
    print(f"Processed: {file_name}")
