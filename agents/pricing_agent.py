from __future__ import annotations
from typing import Optional, Dict, Any

def calculate_premium(
    base_premium: float, 
    fraud_risk_flag: bool,
    medical_risk_class: Optional[str] = "Standard"
) -> Dict[str, Any]:
    """
    Pricing Agent:
    - Base premium: 10000 (passed in)
    - Fraud Load: +25% (0.25)
    - Medical Load:
        - 'Standard': 0% (0.0)
        - 'Substandard': +15% (0.15)
        - 'Decline': +50% (0.50) and suggested manual review
    """
    
    fraud_load = 0.25 if fraud_risk_flag else 0.0
    
    medical_loads = {
        "Standard": 0.0,
        "Substandard": 0.15,
        "Decline": 0.50
    }
    
    # Defaults to Standard if unknown class provided.
    med_load = medical_loads.get(medical_risk_class, 0.0)
    
    # INDUSTRY STANDARD RULE: If intentional misrepresentation is found, DECLINE
    if fraud_risk_flag:
        return {
            "base_premium": float(base_premium),
            "fraud_risk_flag": True,
            "medical_risk_class": medical_risk_class,
            "loads": {"fraud": "FATAL", "medical": med_load},
            "total_load_factor": 1.0, # (Maximum load indicator)
            "premium_total": 0.0,
            "underwriting_decision": "DECLINED (NON-DISCLOSURE DETECTED)"
        }
    
    total_load = fraud_load + med_load
    premium = float(base_premium) * (1.0 + total_load)
    
    return {
        "base_premium": float(base_premium),
        "fraud_risk_flag": bool(fraud_risk_flag),
        "medical_risk_class": medical_risk_class,
        "loads": {
            "fraud": fraud_load,
            "medical": med_load,
        },
        "total_load_factor": total_load,
        "premium_total": premium,
        "underwriting_decision": "Standard" if total_load < 0.1 else ("Substandard" if total_load < 0.4 else "Referred (High Risk)")
    }
