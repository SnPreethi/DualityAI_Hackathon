# search_hyperparams.py
import optuna
from ultralytics import YOLO
import json
import torch

DATA = r"./yolo_params_subset.yaml"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 5

def objective(trial):
    lr = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    mosaic = trial.suggest_float("mosaic", 0.0, 0.8)
    optimizer = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    momentum = trial.suggest_float("momentum", 0.7, 0.97)
    imgsz = trial.suggest_categorical("imgsz", [416, 640, 768])  

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=8,
        imgsz=imgsz,
        lr0=lr,
        optimizer=optimizer,
        momentum=momentum,
        mosaic=mosaic,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    val_map = results.results_dict.get("metrics/mAP50-95", 0.0)
    return val_map

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    best_params = study.best_trial.params
    print("Best trial:", best_params)

    with open("best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Saved best hyperparameters to best_hyperparams.json")
