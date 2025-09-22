# MERGING ORIGINAL AND AUGMENTED DATASETS

# importing libraries
import os, shutil, glob
from pathlib import Path

def merge_folders(src_images, src_labels, dst_images, dst_labels, skip_existing=True):
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    for img in glob.glob(os.path.join(src_images, "*")):
        bn = Path(img).name
        dst_img = os.path.join(dst_images, bn)
        if not (skip_existing and os.path.exists(dst_img)):
            shutil.copy(img, dst_img)
    for lbl in glob.glob(os.path.join(src_labels, "*.txt")):
        bn = Path(lbl).name
        dst_lbl = os.path.join(dst_labels, bn)
        if not (skip_existing and os.path.exists(dst_lbl)):
            shutil.copy(lbl, dst_lbl)
    print(f"Merged {src_images} -> {dst_images}")

if __name__ == "__main__":
    # example usage: merge augmented into original train
    merge_folders("Training_dataset/train1/images", "Training_dataset/train1/labels",
                  "Training_dataset_merged/train1/images", "Training_dataset_merged/train1/labels", skip_existing=False)
    merge_folders("Training_dataset_aug/train1/images", "Training_dataset_aug/train1/labels",
                  "Training_dataset_merged/train1/images", "Training_dataset_merged/train1/labels", skip_existing=False)