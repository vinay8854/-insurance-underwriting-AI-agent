from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional

from google.adk.agents import LlmAgent
from google.genai import types
from pydantic import BaseModel, Field


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HIST_DB_PATH = DATA_DIR / "historical_cases.jsonl"


class KeyMatchVariables(BaseModel):
    # Regulation-safe buckets only (no exact age/address/name).
    age_band: str = Field(..., description="Age band bucket, e.g. '18-30', '40-45', '50+'.")
    gender: Optional[str] = Field(None, description="Gender bucket if available.")
    smoker_status: bool = Field(..., description="True if smoker/nicotine user, else False.")
    bmi_tier: Optional[str] = Field(
        None, description="BMI tier bucket, e.g. 'Normal (18.5-24.9)'."
    )
    primary_diagnosis_icd10: Optional[str] = Field(
        None, description="Primary ICD-10 code if available."
    )
    comorbidities_icd10: List[str] = Field(
        default_factory=list, description="List of comorbidity ICD-10 codes."
    )


class PrecedentFetchInput(BaseModel):
    structured_nlp_json: dict[str, Any] = Field(
        ...,
        description=(
            "Structured JSON output from the NLP/Extraction stage. "
            "May include demographics and extracted medical signals."
        ),
    )
    base_premium_hint: Optional[float] = Field(
        default=None,
        description=(
            "Optional base premium hint from the Pricing Engine. "
            "If provided, the agent must use this exact value when "
            "computing suggested_loaded_premium from historical load percentages."
        ),
    )


class PrecedentFetchOutput(BaseModel):
    fingerprint: KeyMatchVariables = Field(..., description="The derived fingerprint.")
    new_fingerprint_flag: bool = Field(
        ..., description="True if no precedent found in the database."
    )
    matches_found: int = Field(..., description="Number of matching historical cases found.")
    suggested_load_percentage: float = Field(
        ...,
        description=(
            "Suggested premium load percentage based on historical precedents. "
            "0 if no precedent exists."
        ),
    )
    suggested_base_premium: float = Field(
        ...,
        description=(
            "Base premium amount used to compute suggested_loaded_premium. "
            "This is typically provided by the Pricing Engine; "
            "if not available, assume 10000 for illustration."
        ),
    )
    suggested_loaded_premium: float = Field(
        ...,
        description=(
            "Suggested total premium = suggested_base_premium * (1 + suggested_load_percentage/100). "
            "0 if no precedent exists."
        ),
    )
    precedent_summary: str = Field(
        ...,
        description=(
            "Summary for the Lead Underwriter. If no matches, say 'No historical precedent found'."
        ),
    )
    matched_cases: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Small list of matched cases (compressed, non-PII).",
    )


class FinalDecisionInput(BaseModel):
    fingerprint: KeyMatchVariables = Field(..., description="Fingerprint computed in Phase 2.")
    new_fingerprint_flag: bool = Field(..., description="Flag from Phase 2 precedent fetch.")
    final_decision: dict[str, Any] = Field(
        ...,
        description=(
            "Final decision object from Lead Underwriter + Govt Rule Check + Pricing. "
            "Must not include PII."
        ),
    )


class FinalDecisionWriteBackOutput(BaseModel):
    action: str = Field(..., description="Either 'inserted' or 'incremented'.")
    case_id: str = Field(..., description="Case ID that was inserted/updated.")
    match_count: int = Field(..., description="Updated match_count.")
    stored_record_path: str = Field(..., description="Path to the local JSONL database file.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fingerprint_equals(a: KeyMatchVariables, b: dict[str, Any]) -> bool:
    kmv = b.get("key_match_variables") or {}
    return (
        kmv.get("age_band") == a.age_band
        and kmv.get("gender") == a.gender
        and bool(kmv.get("smoker_status")) == bool(a.smoker_status)
        and kmv.get("bmi_tier") == a.bmi_tier
        and kmv.get("primary_diagnosis_icd10") == a.primary_diagnosis_icd10
        and sorted(kmv.get("comorbidities_icd10") or []) == sorted(a.comorbidities_icd10)
    )


def query_historical_cases(fingerprint: dict[str, Any], limit: int = 25) -> dict[str, Any]:
    """
    Native tool: local 'historical_cases' query.
    This replaces Firestore with a local JSONL file in `data/historical_cases.jsonl`.
    """

    fp = KeyMatchVariables.model_validate(fingerprint)
    rows = _load_jsonl(HIST_DB_PATH)
    matches = [r for r in rows if _fingerprint_equals(fp, r)]
    matches = sorted(matches, key=lambda r: r.get("decision_timestamp", ""), reverse=True)[:limit]
    return {"matches": matches, "count": len(matches)}


def store_case_summary(
    fingerprint: dict[str, Any],
    final_decision: dict[str, Any],
    new_fingerprint_flag: bool,
) -> dict[str, Any]:
    """
    Native tool: local write-back.

    Protocol:
    - If new_fingerprint_flag is True -> insert a new row with match_count=1
    - Else -> find existing row and increment match_count

    Stored schema is regulation-safe (no raw age, no PII).
    """

    fp = KeyMatchVariables.model_validate(fingerprint)
    rows = _load_jsonl(HIST_DB_PATH)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    if new_fingerprint_flag:
        case_id = f"HIST-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        row = {
            "case_id": case_id,
            "decision_timestamp": now,
            "key_match_variables": fp.model_dump(exclude_none=True),
            "final_decision": final_decision,
            "match_count": 1,
        }
        _append_jsonl(HIST_DB_PATH, row)
        return {"action": "inserted", "case_id": case_id, "match_count": 1, "path": str(HIST_DB_PATH)}

    # existing fingerprint: increment
    for r in rows:
        if _fingerprint_equals(fp, r):
            r["match_count"] = int(r.get("match_count", 1)) + 1
            r["decision_timestamp"] = now
            _save_jsonl(HIST_DB_PATH, rows)
            return {
                "action": "incremented",
                "case_id": r.get("case_id", "UNKNOWN"),
                "match_count": int(r["match_count"]),
                "path": str(HIST_DB_PATH),
            }

    # Flag said "not new" but no record exists -> recover by inserting.
    case_id = f"HIST-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    row = {
        "case_id": case_id,
        "decision_timestamp": now,
        "key_match_variables": fp.model_dump(exclude_none=True),
        "final_decision": final_decision,
        "match_count": 1,
    }
    _append_jsonl(HIST_DB_PATH, row)
    return {"action": "inserted", "case_id": case_id, "match_count": 1, "path": str(HIST_DB_PATH)}


def create_precedent_fetcher_agent(model: str) -> LlmAgent:
    instruction = """
You are the Historical Case Agent (Phase 2: Precedent Fetcher).

Goal:
From the structured NLP JSON, extract ONLY the regulation-safe Key Match Variables:
- age_band (bucket only; never store exact age)
- gender (if present)
- smoker_status (boolean)
- bmi_tier (bucket)
- primary_diagnosis_icd10
- comorbidities_icd10 (list)

Rules:
- Do NOT use name/address/phone/email or any direct identifier.
- If exact age exists, convert to a bucket (example: 45 -> '40-50'). If already bucketed, keep it.
- If ICD codes are missing, set primary_diagnosis_icd10 = null and comorbidities_icd10 = [].
- If BMI numeric is present, convert to tier buckets:
  Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30)

After you output the fingerprint, call the `query_historical_cases` tool with that fingerprint.
Then:
- Compute the average historical premium_load_percentage across matches (if any).
- Decide suggested_load_percentage as this historical average (rounded sensibly).
- Decide suggested_base_premium:
  * If base_premium_hint is provided in input, you MUST use that exact value.
  * Otherwise, assume 10000 as a neutral base for illustration.
- Compute suggested_loaded_premium = suggested_base_premium * (1 + suggested_load_percentage/100).

Summarize the matches for the Lead Underwriter:
- If matches exist: number of cases, approval rate, average load %, and common exclusions.
- If no matches: 'No historical precedent found' and set new_fingerprint_flag=true.

Output must be valid JSON matching the output schema.
""".strip()

    return LlmAgent(
        model=model,
        name="historical_precedent_fetcher_agent",
        description="Creates fingerprint and fetches historical precedents from local JSONL DB.",
        instruction=instruction,
        include_contents="none",
        tools=[query_historical_cases],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
        input_schema=PrecedentFetchInput,
        output_schema=PrecedentFetchOutput,
        output_key="historical_precedent_result",
    )


def create_precedent_saver_agent(model: str) -> LlmAgent:
    instruction = """
You are the Historical Case Agent (Phase 4: Precedent Saver).

Input:
- fingerprint
- new_fingerprint_flag
- final_decision (already regulation-safe, no PII)

Task:
Call `store_case_summary` tool using these inputs.
Return the tool result as JSON matching the output schema.
""".strip()

    return LlmAgent(
        model=model,
        name="historical_precedent_saver_agent",
        description="Writes back compressed regulation-safe case summaries to local JSONL DB.",
        instruction=instruction,
        include_contents="none",
        tools=[store_case_summary],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
        input_schema=FinalDecisionInput,
        output_schema=FinalDecisionWriteBackOutput,
        output_key="historical_writeback_result",
    )

