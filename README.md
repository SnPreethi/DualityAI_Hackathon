ORDER OF RUNNING SCRIPTS
Phase-1 - Dataset Prep
1. make_subset.py
2. merge_datasets.py
3. aygment_offline.py
4. compute_anchors.py

Phase-2 - Hyperparameter Tuning
5. search_hyperparams.py -> best_hyperparams.json

Phase-3 - Model Training
6. train.py
7. multi_scale_train.py

Phase-4 - Evaluation
8. predict.py
9. visualize.py

Phase-5 - Continuous Learning
10. continuous_daemon.py
11. continuous_train.py