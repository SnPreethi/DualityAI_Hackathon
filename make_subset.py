### SCRIPT TO GENERATE WITH ~200-300 IMAGES FOR FAST EXPERIMENTS
## FOR HYPERPARAM SEARCH

import os, shutil, random

def make_subset(src_img, src_lbl, dst_img, dst_lbl, n=200):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    files = [f for f in os.listdir(src_img) if f.endswith(('.jpg', '.png'))]
    random.shuffle(files)
    for f in files[:n]:
        shutil.copy(os.path.join(src_img, f), os.path.join(dst_img, f))
        lbl = f.rsplit('.',1)[0]+'.txt'
        if os.path.exists(os.path.join(src_lbl, lbl)):
            shutil.copy(os.path.join(src_lbl, lbl), os.path.join(dst_lbl, lbl))

if __name__ == "__main__":
    make_subset("./Training_dataset/train1/images",
                "./Training_dataset/train1/labels",
                "./Training_dataset_subset/train/images",
                "./Training_dataset_subset/train/labels",
                n=200)
    make_subset("./Training_dataset/val1/images",
                "./Training_dataset/val1/labels",
                "./Training_dataset_subset/val/images",
                "./Training_dataset_subset/val/labels",
                n=50)
