from __future__ import annotations

import json
from pathlib import Path


def _load_irdai_rules() -> dict:
    rules_path = Path(__file__).resolve().parents[1] / "data" / "irdai_rules.json"
    with rules_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compliance_check(premium_load_factor: float) -> dict:
    """
    Compliance PoC:
    - Reads `data/irdai_rules.json`
    - Ensures `premium_load_factor <= caps.tobacco_load_max`
    """

    rules = _load_irdai_rules()
    tobacco_load_max = float(rules["caps"]["tobacco_load_max"])

    compliant = float(premium_load_factor) <= tobacco_load_max
    return {
        "premium_load_factor": float(premium_load_factor),
        "tobacco_load_max_cap": tobacco_load_max,
        "compliant": compliant,
    }

