from __future__ import annotations

from typing import List

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field


class FraudInput(BaseModel):
    declared_smoker: bool = Field(
        ..., description="Whether the applicant declared themselves as a smoker."
    )
    tobacco_nicotine_evidence_found: bool = Field(
        ...,
        description="Whether the NLP step found explicit evidence of tobacco or nicotine use in the medical text.",
    )
    evidence_summary: str = Field(
        ..., description="Brief explanation of the evidence found by the NLP step."
    )
    evidence_snippets: List[str] = Field(
        default_factory=list,
        description="Short evidence snippets quoted from the medical text (may be empty).",
    )


class FraudOutput(BaseModel):
    fraud_contradiction: bool = Field(
        ..., description="True if declared_smoker and inferred smoker evidence contradicts underwriting rules."
    )
    fraud_risk_flag: bool = Field(
        ...,
        description="Risk flag derived from fraud_contradiction (for this PoC, it is true when contradiction is true).",
    )
    declared_smoker: bool = Field(..., description="Echo input for traceability.")
    inferred_smoker_from_text: bool = Field(
        ..., description="Inferred smoker status based on evidence found."
    )
    fraud_reason: str = Field(
        ...,
        description="1-2 sentence explanation connecting declared_smoker to the evidence_summary, highlighting why this is flagged.",
    )
    key_evidence_snippets: List[str] = Field(
        default_factory=list,
        description="If available, short snippets that justify the NLP evidence boolean.",
    )


def create_fraud_agent(model: str | LiteLlm) -> LlmAgent:
    instruction = """
You are an Insurance Underwriting fraud/contradiction detection agent.

You will be given:
- declared_smoker: what the applicant stated
- tobacco_nicotine_evidence_found: whether the NLP step found explicit evidence of tobacco/nicotine in medical notes
- evidence_summary: a short description of that evidence
- evidence_snippets: short quoted phrases from the medical text (if available)

Task:
1. Determine inferred_smoker_from_text = tobacco_nicotine_evidence_found
2. Determine fraud_contradiction:
   - fraud_contradiction = true when declared_smoker != inferred_smoker_from_text
3. fraud_risk_flag should be the same as fraud_contradiction for this PoC.
4. Provide fraud_reason that clearly explains the contradiction using evidence_summary.

{
  "fraud_contradiction": boolean,
  "fraud_risk_flag": boolean,
  "declared_smoker": boolean,
  "inferred_smoker_from_text": boolean,
  "fraud_reason": string,
  "key_evidence_snippets": string[]
}
""".strip()

    return LlmAgent(
        model=model,
        name="fraud_contradiction_agent",
        description="Flags contradictions between declared smoker status and evidence of tobacco/nicotine.",
        instruction=instruction,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
        input_schema=FraudInput,
        output_key="fraud_result",
    )

