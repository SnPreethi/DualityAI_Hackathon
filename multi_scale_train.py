"""
Training the model sequentially at multiple image sizes to improve multi-scale detection because the model sees objects at different resolutions.
It uses existing yolov8 weights as initialization and saves best.pt in runs/.
"""

# importing libraries
import os
from ultralytics import YOLO
import json
import argparse

DEFAULT_RUNS_DIR = "runs/detect"

def load_anchor_suggestions(json_path="anchors.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("suggested_img_sizes", [])
    return []

def multi_stage_train(model_name="yolov8s.pt", data_yaml="yolo_params.yaml",
                      stages=(320, 640), epochs_per_stage=(10, 30), batch=8, device="cpu"):
    assert len(stages) == len(epochs_per_stage), "stages and epochs_per_stage must match lengths"

    prev_weights = model_name
    for stage_idx, img_size in enumerate(stages):
        epochs = epochs_per_stage[stage_idx]
        print(f"\n=== Stage {stage_idx+1}/{len(stages)}: imgsz={img_size}, epochs={epochs} ===")
        model = YOLO(prev_weights)
        res = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            device=device,
            save=True
        )
        # settng prev_weights to the best.pt from this run
        prev_weights = res.best
        print(f"Stage {stage_idx+1} finished. Best weights: {prev_weights}")

    print("Multi-stage training finished.")
    return prev_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="Base model (pretrained)")
    parser.add_argument("--data", default="yolo_params.yaml", help="Data YAML path")
    parser.add_argument("--stages", nargs="+", type=int, help="Image sizes list (e.g., 320 640 960)")
    parser.add_argument("--epochs", nargs="+", type=int, help="Epochs per stage (e.g., 5 20 20)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.stages is None:
        # trying to load from anchors.json suggestions
        suggestions = load_anchor_suggestions()
        if suggestions:
            # expanding suggestions to include a middle size if only two
            stages = suggestions
        else:
            stages = [320, 640]
    else:
        stages = args.stages

    if args.epochs is None:
        # default small -> large
        epochs = [5] * len(stages)
        epochs[-1] = 30
    else:
        epochs = args.epochs

    best = multi_stage_train(model_name=args.model, data_yaml=args.data,
                             stages=stages, epochs_per_stage=epochs,
                             batch=args.batch, device=args.device)
    print("Final best weights:", best)


'''
HOW TO RUN
--> python multi_scale_train.py --model yolov8s.pt --data yolo_params.yaml --device cuda:0

'''