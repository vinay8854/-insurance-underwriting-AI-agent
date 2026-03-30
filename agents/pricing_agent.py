from __future__ import annotations


def calculate_premium(base_premium: float, fraud_risk_flag: bool) -> dict:
    """
    Pricing PoC:
    - Base premium: 10000 (passed in)
    - If fraud_risk_flag is true, add a 25% load (0.25)
    """

    load_factor = 0.25 if fraud_risk_flag else 0.0
    premium = float(base_premium) * (1.0 + load_factor)
    return {
        "base_premium": float(base_premium),
        "fraud_risk_flag": bool(fraud_risk_flag),
        "load_factor": load_factor,
        "premium_total": premium,
    }

