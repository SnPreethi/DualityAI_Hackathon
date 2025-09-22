# continuous_daemon.py
import time
import os
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO
from continuous_train import continuous_train

FALCON_DIR = r"Falcon_dataset/images"
FALCON_LABELS = r"Falcon_dataset/labels"
OLD_DATASET = r"Training_dataset/train1"
PREV_MODEL = r"runs/detect/train/weights/best.pt"
CHECK_DELAY = 30
EPOCHS = 10
FALCON_RATIO = 0.3
PRED_DIR = r"Falcon_predictions"

# loading detection model once for inference
model = YOLO(PREV_MODEL)
os.makedirs(PRED_DIR, exist_ok=True)

class FalconHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_update = time.time()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".jpg", ".png", ".jpeg")):
            print(f"New Falcon image detected: {event.src_path}")
            self.last_update = time.time()

    def run_inference(self, img_path):
        results = model.predict(img_path, conf=0.5)
        annotated = results[0].plot()
        fname = os.path.basename(img_path)
        out_path = os.path.join(PRED_DIR, f"pred_{fname}")
        cv2.imwrite(out_path, annotated)
        print(f"Saved prediction: {out_path}")

def run_daemon():
    event_handler = FalconHandler()
    observer = Observer()
    observer.schedule(event_handler, FALCON_DIR, recursive=False)
    observer.start()

    print(f"Watching {FALCON_DIR} for new Falcon data....")
    try:
        while True:
            now = time.time()
            # If new files arrived and idle for CHECK_DELAY → retrain
            if now - event_handler.last_update > CHECK_DELAY:
                # prevent repeat
                event_handler.last_update = float("inf")
                # only if data exists
                if os.listdir(FALCON_DIR):
                    print("New Falcon data ready. Triggering replay-buffer training.")
                    continuous_train(
                        prev_model_path=PREV_MODEL,
                        old_dataset=OLD_DATASET,
                        falcon_dataset="Falcon_dataset",
                        epochs=EPOCHS,
                        falcon_ratio=FALCON_RATIO
                    )
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    run_daemon()
