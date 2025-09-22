# importing libraries
import os
import glob
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse

# Config
LABELS_FOLDER = r'Training_dataset/train1/labels'
IMAGES_FOLDER = r'Training_dataset/train1/images'
# number of anchor clusters
K = 9
OUT_JSON = "best_anchors.json"

def load_bboxes(labels_folder, images_folder):
    wh = []
    sizes = []
    for lf in glob.glob(os.path.join(labels_folder, "*.txt")):
        basename = os.path.basename(lf).rsplit(".", 1)[0]
        possible_imgs = [os.path.join(images_folder, basename + ext) for ext in (".jpg", ".jpeg", ".png")]
        img_path = next((p for p in possible_imgs if os.path.exists(p)), None)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        sizes.append((W, H))
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                pw = bw * W
                ph = bh * H
                if pw < 2 or ph < 2:
                    continue
                wh.append([pw, ph])
    return np.array(wh), sizes

def kmeans_anchors(wh_array, k=9):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(wh_array)
    anchors = km.cluster_centers_
    # sorting by area
    anchors = anchors[np.argsort(anchors[:,0]*anchors[:,1])]
    return anchors

def suggest_img_sizes(anchors, quantiles=(0.1, 0.5, 0.9)):
    # computing areas
    areas = anchors[:,0] * anchors[:,1]
    qvals = np.quantile(areas, quantiles)
    # picking square-ish sizes around sqrt(area)
    sizes = [int(max(32, np.sqrt(q))) for q in qvals]
    sizes = sorted(list(set([((s//32)+1)*32 for s in sizes])))
    return sizes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default=LABELS_FOLDER, help='Path to YOLO label folder')
    parser.add_argument('--images', type=str, default=IMAGES_FOLDER, help='Path to YOLO image folder')
    parser.add_argument('--k', type=int, default=K, help='Number of clusters (anchors)')
    parser.add_argument('--out', type=str, default=OUT_JSON, help='Output JSON file for anchors')
    args = parser.parse_args()

    wh, sizes = load_bboxes(args.labels, args.images)
    if wh.size == 0:
        print("No boxes found; check LABELS_FOLDER and IMAGES_FOLDER paths.")
        exit(1)

    anchors = kmeans_anchors(wh, k=args.k)
    anchors_list = anchors.tolist()

    avg_w = np.mean([s[0] for s in sizes])
    avg_h = np.mean([s[1] for s in sizes])
    avg_img_size = int((avg_w + avg_h) / 2)

    suggested_sizes = suggest_img_sizes(anchors)

    # printing summary
    print("Detected images:", len(sizes))
    print("Average image size (px):", avg_img_size)
    print("\nAnchors (w, h) in pixels (sorted by area):")
    for i, a in enumerate(anchors_list):
        w, h = a
        print(f"{i+1:2d}: w={w:.1f}, h={h:.1f}, area={w*h:.1f}")

    print("\nSuggested multi-scale training image sizes (multiples of 32):", suggested_sizes)

    # saving anchors in JSON
    out = {
        "anchors_px": anchors_list,
        "suggested_img_sizes": suggested_sizes,
        "avg_image_size": avg_img_size,
        "num_images": len(sizes)
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\nSaved anchor info to {args.out}")


'''
RUN COMMAND
python compute_anchors.py --labels Training_dataset/train1/labels --images Training_dataset/train1/images --k 9 --out best_anchors.json
'''