"""Helper methods for interacting with the local model registry via artifacts."""

import glob
import json
from pathlib import Path


def get_production_champion(artifact_root: Path = Path("artifacts/runs")) -> dict | None:
    """Read the latest run.json that successfully passed the quality gate."""
    run_jsons = sorted(glob.glob(str(artifact_root / "*/run.json")), reverse=True)

    for run_json_path in run_jsons:
        try:
            meta = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
            if meta.get("quality_gate", {}).get("passed") is True:
                return meta
        except Exception:
            continue
    return None


def get_champion_validation_mae(artifact_root: Path = Path("artifacts/runs")) -> float | None:
    """Return the validation candidate MAE for the current production champion."""
    champ = get_production_champion(artifact_root)
    if not champ:
        return None
    try:
        return champ["metrics"]["validation"]["candidate"]["mae"]
    except KeyError:
        return None
