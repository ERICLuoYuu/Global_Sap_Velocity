import subprocess

scripts = ["src/hyperparameter_optimization/test_hyperparameter_tuning_ann.py", "src/hyperparameter_optimization/test_hyperparameter_tuning_lstm.py", "src/hyperparameter_optimization/test_hyperparameter_tuning_rf.py", "src/hyperparameter_optimization/test_hyperparameter_tuning_xgb.py"]

# Run sequentially
for script in scripts:
    subprocess.run(["python3", script])


