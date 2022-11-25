"""
- iterate through all sub folders
- iterate through all tif files
- starting at (0, 0), look at img patches of size (224, 224)
- if there is at least 10% of the pixels in channel 3 with value :), extract the patch in channel 1? and save it
- move 224 pixels over, continue until all imgs are done
"""

import os
import tifffile
import numpy as np
from matplotlib import pyplot as plt


data_path = "./data/Dataset"
all_subdirs = [x[0] for x in os.walk(data_path)]
output_path = "./data/factin_patches"
if not os.path.exists(output_path):
    os.mkdir(output_path)
patches_size = 224

saved_patches_idx = 0
break_flag = False
for subdir in all_subdirs:

    img_files = [
        f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and f.split(".")[-1] == "tif"
    ]

    for img_name in img_files:

        print(f"Extracting patches from img {img_name} in {subdir}")
        img = tifffile.imread(subdir + "/" + img_name)

        for row in range(0, img.shape[1], patches_size):
            for col in range(0, img.shape[2], patches_size):

                foreground = img[-1, row: row+patches_size, col: col+patches_size]
                foreground = (foreground - np.min(foreground)) / (np.max(foreground) - np.min(foreground))

                patch = img[0, row: row+patches_size, col: col+patches_size]
                if patch.shape[0] != patches_size or patch.shape[1] != patches_size:
                    continue

                foreground_thresholded = np.where(foreground > 0.25, 1, foreground)

                percentage_foreground = np.sum(foreground_thresholded) / patches_size ** 2
                if percentage_foreground >= 0.1:

                    patch_normalized = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
                    np.savez(output_path + f"/{saved_patches_idx}.npz", patch_normalized)

                    saved_patches_idx += 1

            if break_flag:
                break
        if break_flag:
            break
    if break_flag:
        break

print(":)")
