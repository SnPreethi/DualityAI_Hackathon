# importing libraries
import os
import glob
import cv2
import random
import argparse
import albumentations as A
from pathlib import Path
from albumentations import (
    RandomBrightnessContrast,
    RandomGamma,
    CLAHE,
    MotionBlur,
    MedianBlur,
    Affine,
    Perspective,
    GaussNoise,
    CoarseDropout,
)

def read_yolo_labels(label_path):
    bboxes = []
    class_ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            bboxes.append([cx, cy, w, h])
            class_ids.append(cls)
    return bboxes, class_ids

def write_yolo_labels(label_path, bboxes, class_ids):
    with open(label_path, "w") as f:
        for (cx, cy, w, h), cid in zip(bboxes, class_ids):
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_xyxy(bbox, img_w, img_h):
    cx, cy, w, h = bbox
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return [x1, y1, x2, y2]

def xyxy_to_yolo(xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w/2
    cy = y1 + h/2
    return [cx/img_w, cy/img_h, w/img_w, h/img_h]

def create_transforms(img_size):
    return A.Compose([
        RandomBrightnessContrast(p=0.5),
        Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.05), rotate=(-15, 15), shear=(-5, 5), p=0.7, cval=0, cval_mask=0),
        A.OneOf([
            MotionBlur(blur_limit=5),
            MedianBlur(blur_limit=5),
            GaussNoise(var_limit=(10.0, 50.0))
        ], p=0.3),
        A.OneOf([
            RandomGamma(),
            CLAHE()
        ], p=0.3),
        Perspective(scale=(0.02, 0.1), p=0.3),
        CoarseDropout(max_holes=1, max_height=int(img_size*0.08), max_width=int(img_size*0.08), p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def augment_dataset(src_images, src_labels, dst_images, dst_labels, n_augment=3, img_size=640, random_seed=0):
    random.seed(random_seed)
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    transform = create_transforms(img_size)

    img_files = [p for p in glob.glob(os.path.join(src_images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(img_files)} images to augment.")

    for img_path in img_files:
        base = Path(img_path).stem
        label_path = os.path.join(src_labels, base + ".txt")
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bboxes_yolo, class_ids = read_yolo_labels(label_path)
        if len(bboxes_yolo) == 0:
            continue

        # Convert to pascal_voc for Albumentations
        bboxes_xy = [yolo_to_xyxy(b, w, h) for b in bboxes_yolo]
        for i in range(n_augment):
            try:
                augmented = transform(image=img, bboxes=bboxes_xy, class_labels=class_ids)
            except Exception as e:
                print("Transform failed:", e)
                continue
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            # filter boxes: remove very small boxes
            good_boxes = []
            good_labels = []
            for bb, cid in zip(aug_bboxes, aug_labels):
                x1, y1, x2, y2 = bb
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 8 or box_h < 8:
                    continue
                # clip
                if x2 <= x1 or y2 <= y1:
                    continue
                good_boxes.append(bb)
                good_labels.append(cid)
            if len(good_boxes) == 0:
                # skip this augmented sample (no valid boxes)
                continue
            # Resize/pad image to img_size x img_size while preserving boxes if needed
            # Let albumentations handle shapes; we will save as-is (but optionally resize)
            out_img_name = f"{base}_aug{i}.jpg"
            out_img_path = os.path.join(dst_images, out_img_name)
            cv2.imwrite(out_img_path, aug_img)

            # convert boxes back to YOLO normalized
            H2, W2 = aug_img.shape[:2]
            yolo_boxes = [xyxy_to_yolo(bb, W2, H2) for bb in good_boxes]
            out_lbl_path = os.path.join(dst_labels, f"{base}_aug{i}.txt")
            write_yolo_labels(out_lbl_path, yolo_boxes, good_labels)

    print("Augmentation finished. Augmented data saved to:", dst_images, dst_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_images", default="Training_dataset/train1/images")
    parser.add_argument("--src_labels", default="Training_dataset/train1/labels")
    parser.add_argument("--dst_images", default="Training_dataset_aug/train1/images")
    parser.add_argument("--dst_labels", default="Training_dataset_aug/train1/labels")
    parser.add_argument("--n", type=int, default=3, help="augmentations per image")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    augment_dataset(args.src_images, args.src_labels, args.dst_images, args.dst_labels,
                    n_augment=args.n, img_size=args.img_size)



'''
HOW TO RUN SCRIPT
python augment_offline.py --n 3 --img_size 640
'''