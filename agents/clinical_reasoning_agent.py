from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import Any, List, Dict, Optional

import httpx
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field

# --- Schemas ---

class ClinicalReasoningInput(BaseModel):
    contexts: List[Dict[str, Any]] = Field(..., description="Raw drug data collected from RxNorm and openFDA.")

class ClinicalDrugAssessment(BaseModel):
    drug_name: str = Field(..., description="The name of the drug evaluated.")
    rxcui: Optional[str] = Field(None, description="The standardized RxNorm ID.")
    standard_name: Optional[str] = Field(None, description="The normalized name from RxNorm.")
    common_side_effects: List[str] = Field(default_factory=list, description="Common side effects (e.g., nausea).")
    serious_boxed_warnings: List[str] = Field(default_factory=list, description="Severe FDA boxed warnings.")
    underwriting_logic: str = Field(..., description="Reasoning for how this drug affects insurance risk.")

class ClinicalReasoningOutput(BaseModel):
    assessments: List[ClinicalDrugAssessment] = Field(default_factory=list, description="List of risk assessments for each detected drug.")
    overall_medical_risk_class: str = Field(..., description="Aggregated risk classification (e.g., 'Standard', 'Substandard', 'Decline').")
    clinical_summary: str = Field(..., description="Final clinical overview of the applicant's medical logic.")

# --- API Helpers ---

async def get_rxcui(drug_name: str) -> Optional[Dict[str, str]]:
    """Normalizes drug name to RxCUI using RxNorm API."""
    url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={drug_name}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                concept_groups = data.get("drugGroup", {}).get("conceptGroup", [])
                for group in concept_groups:
                    props = group.get("conceptProperties", [])
                    if props:
                        return {
                            "rxcui": props[0].get("rxcui"),
                            "name": props[0].get("name")
                        }
        except Exception:
            pass
    return None

async def get_fda_label_data(drug_name: str) -> Dict[str, Any]:
    """Gets official labeling data from openFDA."""
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    res = results[0]
                    return {
                        "boxed_warning": res.get("boxed_warning", []),
                        "adverse_reactions": res.get("adverse_reactions", []),
                        "warnings": res.get("warnings", [])
                    }
        except Exception:
            pass
    return {}

async def get_fda_event_signal(drug_name: str) -> List[str]:
    """Gets side effect frequency signal from openFDA events."""
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&count=patient.reaction.reactionmeddrapt.exact"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                results = response.json().get("results", [])
                # Return top 5 reactions
                return [r.get("term") for r in results[:5]]
        except Exception:
            pass
    return []

# --- Local Cache Store (Demo Firebase) ---

class PharmacyCache:
    def __init__(self, cache_file: str = "pharmacy_cache.json"):
        self.cache_path = Path(cache_file)
        if not self.cache_path.exists():
            self.cache_path.write_text(json.dumps({}))
        
    def get(self, drug_name: str) -> Optional[Dict[str, Any]]:
        with self.cache_path.open("r") as f:
            cache = json.load(f)
            return cache.get(drug_name.lower())

    def set(self, drug_name: str, data: Dict[str, Any]):
        with self.cache_path.open("r") as f:
            cache = json.load(f)
        cache[drug_name.lower()] = data
        with self.cache_path.open("w") as f:
            json.dump(cache, f, indent=2)

cache_store = PharmacyCache()

# --- Agent Creation ---

def create_clinical_reasoning_agent(model: str | LiteLlm) -> LlmAgent:
    instruction = """
You are a Clinical Reasoning & Pharmacology Risk Underwriting Agent.

Your goal is to evaluate the true medical risk of an applicant based on their medications and FDA data.

Workflow:
1. You will receive a list of drugs and their raw FDA/RxNorm data (if found).
2. For each drug, translate the medical jargon into a clear underwriting risk summary.
3. Specifically look for:
   - Contraindications (Boxed warnings)
   - Underlying conditions the drug suggests (e.g., Metformin -> Diabetes)
   - Frequency of adverse events
4. Assign an overall medical risk class to the applicant:
   - 'Standard': Low risk, maintenance drugs for minor issues.
   - 'Substandard': Chronic conditions with manageable risk.
   - 'Decline': High-risk drugs or serious boxed warnings found5. Output must be raw JSON (no markdown fences, no text outside the JSON).

JSON Keys to return:
{
 "assessments": [
    {
      "drug_name": string,
      "standard_name": string,
      "common_side_effects": string[],
      "serious_boxed_warnings": string[],
      "underwriting_logic": string
    }
 ],
 "overall_medical_risk_class": "Standard" | "Substandard" | "Decline",
 "clinical_summary": string
}
""".strip()

    return LlmAgent(
        model=model,
        name="clinical_reasoning_agent",
        description="Evaluates pharmacology risk using RxNorm and openFDA data.",
        instruction=instruction,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
        ),
        input_schema=ClinicalReasoningInput,
        output_key="clinical_risk_result"
    )

# --- Main Logic Function ---

async def evaluate_clinical_risk(drug_names: List[str], model: str | LiteLlm) -> ClinicalReasoningOutput:
    assessments_data = []
    
    for name in drug_names:
        # Step 1: Cache Check
        cached = cache_store.get(name)
        if cached:
            assessments_data.append(cached)
            continue
            
        # Step 2: RxNorm Normalization
        norm = await get_rxcui(name)
        rxcui = norm.get("rxcui") if norm else None
        standard_name = norm.get("name") if norm else name
        
        # Step 3: FDA Risk Extraction
        label_data = await get_fda_label_data(standard_name)
        event_data = await get_fda_event_signal(standard_name)
        
        # Prepare context for LLM
        drug_context = {
            "drug_name": name,
            "standard_name": standard_name,
            "rxcui": rxcui,
            "fda_label": label_data,
            "common_reactions": event_data
        }
        assessments_data.append(drug_context)

    # Use LLM to reason over all drugs found
    agent = create_clinical_reasoning_agent(model)
    
    # We'll use a simple mock runner or just trigger the agent logic
    # In a real ADK setup, we'd use runner.run_async. 
    # For this implementation, the main underwriting session will handle it.
    # Here we return the raw data collected to be passed to the LLM step in main.py
    return assessments_data
