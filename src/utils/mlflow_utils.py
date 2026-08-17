"""Shared MLflow experiment tracking helpers."""

import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

MLRUNS_DIR = Path("mlruns").resolve()

EXPERIMENTS = {
    "module1": "module1_cold_start",
    "module2": "module2_monetization",
    "module3": "module3_competing_risks",
}


@contextmanager
def tracked_run(module: str, run_name: str):
    """Context manager that points MLflow at the local mlruns/ dir and opens
    a run under the given module's experiment."""
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENTS[module])
    with mlflow.start_run(run_name=run_name) as run:
        yield run
