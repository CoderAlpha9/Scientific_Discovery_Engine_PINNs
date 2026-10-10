import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def relative_error(prediction, reference):
    return float(np.linalg.norm(np.asarray(prediction) - reference) / max(np.linalg.norm(reference), 1e-12))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
