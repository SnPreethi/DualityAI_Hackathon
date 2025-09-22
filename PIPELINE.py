# pipeline.py
import os
import subprocess

def run(cmd):
    print(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}")

if __name__ == "__main__":
    # Phase 1: Dataset Prep
    run("python3 make_subset.py")
    run("python3 merge_datasets.py")
    run("python3 augment_offline.py")
    run("python3 compute_anchors.py")

    # Phase 2: Hyperparameter Tuning
    run("python3 search_hyperparams.py")

    # Phase 3: Training
    run("python3 train.py --use_best --epochs 50")
    run("python3 multi_scale_train.py")

    # Phase 4: Evaluation
    run("python3 predict.py")
    run("python3 visualize.py")

    # Phase 5: Continuous Learning
    print("\nStarting Continuous Daemon (Ctrl+C to stop)...\n")
    run("python3 continuous_daemon.py")