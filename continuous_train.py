# importing libraries
import os
import random
import shutil
import torch
from ultralytics import YOLO
from pathlib import Path

# Building a temporary replay dataset that mixes old + Falcon data at a given ratio
def build_replay_dataset(old_dataset, falcon_dataset, out_dataset, falcon_ratio=0.3):
    out_img = Path(out_dataset) / "images"
    out_lbl = Path(out_dataset) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    old_imgs = list((Path(old_dataset) / "images").glob("*"))
    falcon_imgs = list((Path(falcon_dataset) / "images").glob("*"))

    old_labels = {p.stem: p for p in (Path(old_dataset) / "labels").glob("*.txt")}
    falcon_labels = {p.stem: p for p in (Path(falcon_dataset) / "labels").glob("*.txt")}

    total = min(len(old_imgs) + len(falcon_imgs), 2000)  # cap for sanity
    falcon_count = int(total * falcon_ratio)
    old_count = total - falcon_count

    chosen_old = random.sample(old_imgs, min(old_count, len(old_imgs)))
    chosen_falcon = random.choices(falcon_imgs, k=falcon_count)  # allowing repetition

    # copying images + labels into replay buffer
    for img_path in chosen_old + chosen_falcon:
        stem = img_path.stem
        if "falcon" in img_path.as_posix():
            tag = "falcon"
        else:
            tag = "old"
        new_name = f"{tag}_{stem}{img_path.suffix}"
        new_img_path = out_img / new_name
        shutil.copy(img_path, new_img_path)

        # copying label
        lbl_path = old_labels.get(stem) if tag == "old" else falcon_labels.get(stem)
        if lbl_path and lbl_path.exists():
            new_lbl_path = out_lbl / f"{new_name.rsplit('.',1)[0]}.txt"
            shutil.copy(lbl_path, new_lbl_path)

    print(f"Replay buffer built at {out_dataset}")
    print(f"Old samples: {len(chosen_old)}, Falcon samples: {len(chosen_falcon)}")

# fine-tuning trained YOLO model with new Falcon data
def continuous_train(prev_model_path, old_dataset, falcon_dataset, epochs=20, falcon_ratio=0.3):

    replay_dir = 'Replay_dataset'
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    build_replay_dataset(old_dataset, falcon_dataset, replay_dir, falcon_ratio)

    # building dataset YAML for replay buffer
    yaml_file = "yolo_params_replay.yaml"
    with open(yaml_file, "w") as f:
        f.write(f"train: {replay_dir}\n")
        f.write(f"val: {old_dataset.replace('train1','val1')}\n")  # reuse validation
        f.write("nc: 7\n")
        f.write("names: ['OxygenTank','NitrogenTank','FirstAidBox','FireAlarm','SafetySwitchPanel','EmergencyPhone','FireExtinguisher']\n")


    # loadig model and resuming from previous best
    model = YOLO(prev_model_path)

    # fine-tune
    results = model.train(
        data=yaml_file,
        epochs=epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume=False,
        batch=16,
        imgsz=640,
        mosaic=0.5,
        lr0=0.00005,
    )

    print(f'Continuous learning complete. Results saved to {results.save_dir}')
    return results

if __name__ == "__main__":
    prev_model = r'runs/detect/train/weights/best.pt'
    old_dataset = r'Training_dataset/train1'
    falcon_dataset = r'Falcon_dataset'
    continuous_train(prev_model, old_dataset, falcon_dataset, epochs=15, falcon_ratio=0.3)
