from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field

# --- Schemas ---

class PolicyGenerationInput(BaseModel):
    applicant_name: str = Field(..., description="Full name of the applicant.")
    underwriting_decision: str = Field(..., description="The final decision (e.g., 'Standard', 'Substandard', 'Referred').")
    total_premium: float = Field(..., description="The final calculated insurance premium.")
    risk_logic: str = Field(..., description="Actuarial reasoning for the load (from Clinical/NLP agents).")
    total_load_factor: float = Field(..., description="The total surcharge factor (e.g., 0.15).")

class PolicyGenerationOutput(BaseModel):
    policy_body_text: str = Field(..., description="The formal, polite, customer-facing policy summary or rejection message.")
    document_title: str = Field(..., description="Title for the document (e.g., 'Policy Schedule' or 'Adverse Action Letter').")
    next_steps: str = Field(..., description="Next steps for the applicant.")

# --- Agent Creation ---

def create_policy_generation_agent(model: str | LiteLlm) -> LlmAgent:
    instruction = """
You are a Policy Issuance & Customer Communication Agent.

Your task is to translate robotic underwriting data into a professional, legally-compliant, and polite insurance document.

Steps:
1. Routing:
   - If decision is 'Standard' (Load is 0): Use a welcoming, congratulatory tone.
   - If decision is 'Substandard' (Load > 0): Explain the 'premium adjustment' politely. Use terms like 'medical history evaluation' instead of 'diabetes load'.
   - If decision is 'Referred' or 'Decline': Be professional, empathetic, and clear about the next steps (manual review).

2. Summary Generation:
   - Include the applicant's name and the total premium clearly.
   - Summarize the reasoning provided in 'risk_logic' in a way that is easy for a human to understand but remains accurate.

3. Output rules:
   - MUST output ONLY a valid JSON object.
   - NO markdown fences.
   - Fields to include: "policy_body_text", "document_title", "next_steps".
   - The 'policy_body_text' should be formatted with newlines and look like a formal letter.
""".strip()

    return LlmAgent(
        model=model,
        name="policy_generation_agent",
        description="Generates human-readable insurance policy summaries and letters.",
        instruction=instruction,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        ),
        input_schema=PolicyGenerationInput,
        output_key="policy_generation_result"
    )

# --- Issuance Logic ---

def generate_policy_text_file(applicant_name: str, policy_data: Dict[str, Any]) -> str:
    """Creates a .txt file for the policy demo."""
    safe_name = applicant_name.replace(" ", "_").lower()
    file_path = Path(f"policy_{safe_name}.txt")
    
    # Use .get() with defaults to prevent KeyErrors if the AI response is slightly off
    title = policy_data.get('document_title', 'Insurance Policy Summary')
    body = policy_data.get('policy_body_text', 'Status: Final Underwriting evaluation completed.')
    steps = policy_data.get('next_steps', 'Contact your broker for more details.')

    content = f"""
==============================================================================
{title}
==============================================================================

{body}

------------------------------------------------------------------------------
NEXT STEPS:
{steps}
------------------------------------------------------------------------------
Generated at: {os.uname().nodename if hasattr(os, 'uname') else 'LOCAL-ISSUANCE-NODE'}
Status: ISSUED-TO-BROKER
==============================================================================
"""
    file_path.write_text(content.strip(), encoding="utf-8")
    return str(file_path.resolve())
