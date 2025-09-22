'''# RUNNING THE SCRIPT
python train.py --use_best --epochs 50'''

# importing libraries
import argparse
from ultralytics import YOLO
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random
import cv2

# DEFAULT HYPERPARAMETERS
DEFAULT_PARAMS = {
    'epochs': 10,
    'mosaic': 0.4,
    'optimizer': 'AdamW',
    'momentum': 0.9,
    'lr0': 0.0001,
    'lrf': 0.0001,
    'single_cls': False,
    'imgsz': 640,
}

def load_best_params(json_file="best_hyperparams.json"):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            best_params = json.load(f)
        print(f'Loaded best hyperparameters from {json_file}: {best_params}')
        # Merge defaults + best params (best overrides defaults)
        return {**DEFAULT_PARAMS, **best_params}
    else:
        print('No best_hyperparams.json found. Using defaults.')
        return DEFAULT_PARAMS
    
def load_custom_anchors(anchor_file='best_anchors.json'):
    if os.path.exists(anchor_file):
        with open(anchor_file, 'r') as f:
            anchor_data = json.load(f)
        anchors = anchor_data.get('anchors_px', None)
        suggested_sizes = anchor_data.get('suggested_img_sizes', [])
        print(f'Loaded custom anchors from {anchor_file}')
        print(f'Suggested image sizes: {suggested_sizes}')
        return anchors, suggested_sizes
    else:
        print('No custom anchors found. Using default YOLO anchors.')
        return None, []

def plot_training_curves(run_dir, data_yaml):
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print('No results.csv found to plot curves')
        return

    df = pd.read_csv(results_csv)

    # LOSS CURVE
    plt.figure()
    plt.plot(df["epoch"].to_numpy(), df["train/box_loss"].to_numpy(), label="Box Loss")
    plt.plot(df["epoch"].to_numpy(), df["train/cls_loss"].to_numpy(), label="Class Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"].to_numpy(), df["train/dfl_loss"].to_numpy(), label="DFL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    # METRICS CURVE
    plt.figure()
    plt.plot(df["epoch"].to_numpy(), df["metrics/precision(B)"].to_numpy(), label="Precision")
    plt.plot(df["epoch"].to_numpy(), df["metrics/recall(B)"].to_numpy(), label="Recall")
    plt.plot(df["epoch"].to_numpy(), df["metrics/mAP50(B)"].to_numpy(), label="mAP@0.5")
    plt.plot(df["epoch"].to_numpy(), df["metrics/mAP50-95(B)"].to_numpy(), label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "metrics_curve.png"))
    plt.close()

    print(f"Saved training curves in {run_dir} as loss_curve.png and metrics_curve.png")

    # PER-CLASS AP + CONFUSION MATRIX
    print(f'Running final validation to get per-class metrics.')
    val_results = model.val(data=data_yaml, split="val")
    
    # Per-class AP
    names = val_results.names
    ap_per_class = val_results.results_dict.get("metrics/class_ap", None)

    if ap_per_class is not None:
        plt.figure(figsize=(10,6))
        plt.bar(range(len(names)), ap_per_class, tick_label=list(names.values()))
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("AP@0.5:0.95")
        plt.title("Per-Class Average Precision")
        plt.grid(axis="y")
        plt.savefig(os.path.join(run_dir, "per_class_ap.png"))
        plt.close()
        print(f"Saved per-class AP plot as per_class_ap.png in {run_dir}")
    else:
        print("Could not extract per-class AP. Check Ultralytics version.")

    # CONFUSION MATRIX
    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        cm = val_results.confusion_matrix.matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=list(names.values()),
                    yticklabels=list(names.values()))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
        plt.close()
        print(f"Saved confusion matrix as confusion_matrix.png in {run_dir}")
    else:
        print("Confusion matrix not available. Try upgrading ultralytics.")

def save_sample_predictions(model, images_dir, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("No images found for sample predictions.")
        return

    # Pick random images
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    for img_name in sample_files:
        img_path = os.path.join(images_dir, img_name)
        results = model.predict(img_path, conf=0.5)
        annotated = results[0].plot()

        save_path = os.path.join(output_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, annotated)
        print(f"Saved prediction: {save_path}")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_best', action='store_true', help='Use best_hyperparams.json if available')
    parser.add_argument('--epochs', type=int, default=DEFAULT_PARAMS["epochs"], help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=DEFAULT_PARAMS["mosaic"], help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_PARAMS["optimizer"], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=DEFAULT_PARAMS["momentum"], help='Momentum')
    parser.add_argument('--lr0', type=float, default=DEFAULT_PARAMS["lr0"], help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=DEFAULT_PARAMS["lrf"], help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=DEFAULT_PARAMS["single_cls"], help='Single class training')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_PARAMS["imgsz"], help='Image size')
    args = parser.parse_args()

    # Load best params if flag is set
    if args.use_best:
        params = load_best_params()
    else:
        # using CLI args as dict
        params = vars(args)

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # Injecting custom anchors
    anchors, suggested_sizes = load_custom_anchors(os.path.join(this_dir, "best_anchors.json"))
    if anchors is not None:
        if hasattr(model.model, "model"):
            detect_layer = model.model.model[-1]
            if hasattr(detect_layer, "anchors"):
                detect_layer.anchors = torch.tensor(anchors, dtype=torch.float32)
                print("Custom anchors successfully injected into YOLO model.")
            else:
                print("Could not inject anchors (Detect layer not found).")
        else:
            print("Model structure not standard. Skipping anchor injection.")

    # Multi-scale training using suggested sizes
    if suggested_sizes:
        params['imgsz'] = suggested_sizes
        print(f"Using multi-scale training image sizes: {params['imgsz']}")
    
    # full training call
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=params["epochs"],
        batch=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        single_cls=params["single_cls"], 
        mosaic=params["mosaic"],
        optimizer=params["optimizer"], 
        lr0=params["lr0"], 
        lrf=params["lrf"], 
        momentum=params["momentum"],
        imgsz=params["imgsz"],
        multi_scale = True
    )

    # YOLO save results
    run_dir = results.save_dir
    data_yaml = os.path.join(this_dir, 'yolo_params.yaml')
    plot_training_curves(run_dir, data_yaml)

    # Saving sample predictions
    pred_dir = os.path.join(run_dir, 'predictions')
    val_images_dir = os.path.join(this_dir, 'Training_dataset/val1/images')
    save_sample_predictions(model, val_images_dir, pred_dir, num_samples=5)

'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''