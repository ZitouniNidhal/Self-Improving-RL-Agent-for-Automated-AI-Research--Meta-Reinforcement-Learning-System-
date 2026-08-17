from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict

import numpy as np


class ExperimentLogger:
    def __init__(self, log_dir: str = "runs/default"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_file = None
        self._writer = None
        self._fieldnames = None
        self.start_time = time.time()

    def log(self, generation: int, metrics: Dict[str, Any]) -> None:
        row = {"generation": generation, "elapsed_s": round(time.time() - self.start_time, 2), **metrics}

        if self._writer is None or any(k not in self._fieldnames for k in row):
            existing_rows = []
            if self._writer is not None:
                self._csv_file.close()
                with open(self.csv_path, "r", newline="") as f:
                    existing_rows = list(csv.DictReader(f))
            self._fieldnames = list(dict.fromkeys((self._fieldnames or []) + list(row.keys())))
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            for old_row in existing_rows:
                self._writer.writerow(old_row)

        self._writer.writerow(row)
        self._csv_file.flush()

        pretty = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in metrics.items())
        print(f"[gen {generation:04d}] {pretty}")

    def save_checkpoint(self, policy, name: str = "policy.npy") -> str:
        path = os.path.join(self.log_dir, name)
        np.save(path, policy.get_flat_params())
        return path

    def load_checkpoint(self, policy, name: str = "policy.npy") -> None:
        path = os.path.join(self.log_dir, name)
        flat = np.load(path)
        policy.set_flat_params(flat)

    def save_config(self, config: Dict[str, Any]) -> None:
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
