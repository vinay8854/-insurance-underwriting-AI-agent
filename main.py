from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
from google.genai import types

# Ensure local imports work when running `python main.py` from elsewhere.
sys.path.append(str(Path(__file__).resolve().parent))

from agents import create_nlp_agent, create_fraud_agent
from agents.compliance_agent import compliance_check
from agents.pricing_agent import calculate_premium


APP_NAME = "insurance_adk_demo"
USER_ID = "demo_user_001"
SESSION_ID = "underwriting_session_001"


def _find_json_object(text: str) -> dict[str, Any]:
    """
    Attempts to extract the first JSON object found in the model output.
    ADK output_schema *should* already return pure JSON, but this is a safety net.
    """

    candidate = text.strip()
    # Strip common markdown fences if the model included them.
    if candidate.startswith("```"):
        # Take the content inside the first fenced block.
        parts = candidate.split("```", 2)
        if len(parts) >= 2:
            candidate = parts[1].strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the LLM output.")

    json_str = candidate[start : end + 1]
    return json.loads(json_str)


def _card(title: str) -> None:
    bar = "=" * 78
    print(bar)
    print(f"{title}")
    print(bar)


def _subcard(title: str) -> None:
    bar = "-" * 78
    print(bar)
    print(title)
    print(bar)


async def run_llm_agent(
    agent: LlmAgent,
    runner: Runner,
    session_service: InMemorySessionService,
    session_id: str,
    *,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    input_json = json.dumps(input_data)
    user_message = types.Content(role="user", parts=[types.Part(text=input_json)])

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=user_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None) and part.text.strip():
                    final_text = part.text.strip()
                    break
        # Optional: stream events for deeper debugging
        # print(f"DEBUG event: {event}")

    if not final_text:
        raise RuntimeError(f"No final response text captured from agent '{agent.name}'.")

    try:
        parsed = _find_json_object(final_text)
    except Exception as e:
        raise RuntimeError(
            f"Failed parsing JSON output for agent '{agent.name}'. Raw output: {final_text}"
        ) from e

    # Also attempt to read what ADK stored in session state.
    # This isn't required for correctness (we parse `final_text`), but helps verification.
    try:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        if session and agent.output_key:
            _ = session.state.get(agent.output_key)
    except Exception:
        pass

    return parsed


async def main() -> None:
    base_dir = Path(__file__).resolve().parent
    # Load `insurance_adk_demo/.env` for local runs (keeps project self-contained).
    load_dotenv(dotenv_path=base_dir / ".env")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set `GOOGLE_API_KEY` (or `GOOGLE_GENAI_API_KEY`) in a local .env file."
        )

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not gemini_model:
        raise RuntimeError("Missing GEMINI_MODEL. Set it in `.env` (default is gemini-2.5-flash).")
    dummy_payload_path = base_dir / "data" / "dummy_payload.json"
    with dummy_payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Rules are read inside compliance_agent.py, but we also print them for visibility.
    _rules_path = base_dir / "data" / "irdai_rules.json"
    with _rules_path.open("r", encoding="utf-8") as f:
        rules = json.load(f)

    _card("Local Proof of Concept: Health Insurance Underwriting (ADK + Gemini)")
    print("Execution mode: all local memory | LLM steps: NLP + Fraud | Math steps: Pricing + Compliance")
    print()
    _subcard("Input Payload (Hardcoded)")
    print(json.dumps(payload, indent=2))
    print()
    _subcard("IRDAI Rules (Hardcoded)")
    print(json.dumps(rules, indent=2))

    # Session and runners (in-memory, purely local)
    session_service = InMemorySessionService()

    # Step A: NLP extraction
    nlp_input = {"raw_medical_text": payload["raw_medical_text"]}
    nlp_result = None
    used_nlp_model = None

    _subcard("Step A: NLP Evidence Extraction (Gemini)")
    attempt_session_id = f"{SESSION_ID}_nlp"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=attempt_session_id
    )
    nlp_agent = create_nlp_agent(model=gemini_model)
    nlp_runner = Runner(agent=nlp_agent, app_name=APP_NAME, session_service=session_service)
    nlp_result = await run_llm_agent(
        nlp_agent,
        nlp_runner,
        session_service,
        attempt_session_id,
        input_data=nlp_input,
    )
    used_nlp_model = gemini_model

    if not nlp_result or used_nlp_model is None:
        raise RuntimeError(
            "NLP step failed for all Gemini model candidates due to quota/errors."
        )
    print("Medical text analyzed:")
    print(f"  {payload['raw_medical_text']}")
    print()
    print("NLP output:")
    print(json.dumps(nlp_result, indent=2))

    # Step B: Fraud/contradiction
    fraud_input = {
        "declared_smoker": payload["declared_smoker"],
        "tobacco_nicotine_evidence_found": nlp_result[
            "tobacco_nicotine_evidence_found"
        ],
        "evidence_summary": nlp_result["evidence_summary"],
        "evidence_snippets": nlp_result.get("evidence_snippets", []),
    }

    fraud_result = None
    used_fraud_model = None

    _subcard("Step B: Fraud Contradiction Detection (Gemini)")
    attempt_session_id = f"{SESSION_ID}_fraud"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=attempt_session_id
    )
    fraud_agent = create_fraud_agent(model=gemini_model)
    fraud_runner = Runner(agent=fraud_agent, app_name=APP_NAME, session_service=session_service)
    fraud_result = await run_llm_agent(
        fraud_agent,
        fraud_runner,
        session_service,
        attempt_session_id,
        input_data=fraud_input,
    )
    used_fraud_model = gemini_model
    print("Fraud inputs:")
    print(json.dumps(fraud_input, indent=2))
    print()
    print("Fraud output:")
    print(json.dumps(fraud_result, indent=2))

    # Step C: Pricing
    _subcard("Step C: Pricing Load (Pure Python)")
    base_premium = 10000
    pricing = calculate_premium(
        base_premium=base_premium, fraud_risk_flag=fraud_result["fraud_risk_flag"]
    )
    print(json.dumps(pricing, indent=2))

    # Step D: Compliance
    _subcard("Step D: Compliance Check (Pure Python)")
    compliance = compliance_check(premium_load_factor=pricing["load_factor"])
    print(json.dumps(compliance, indent=2))

    # Final report
    _subcard("Decision Summary")
    tobacco_evidence = nlp_result["tobacco_nicotine_evidence_found"]
    declared_smoker = payload["declared_smoker"]
    inferred_smoker = fraud_result["inferred_smoker_from_text"]
    fraud_flag = fraud_result["fraud_risk_flag"]

    print(f"Declared smoker: {declared_smoker}")
    print(f"Evidence of tobacco/nicotine in note: {tobacco_evidence}")
    print(f"Inferred smoker from evidence: {inferred_smoker}")
    print(f"Fraud contradiction risk flag: {fraud_flag}")
    print(f"Applied premium load: {pricing['load_factor']}")
    print(
        f"Compliant load (<= {compliance['tobacco_load_max_cap']}): {compliance['compliant']}"
    )

    # Make it explicit how the model reasoned (without dumping raw chain-of-thought).
    _subcard("Why the Model Flagged It (LLM Reasoning)")
    print("NLP reasoning summary:")
    print(nlp_result.get("reasoning_summary", "(no reasoning_summary returned)"))
    print()
    print("Fraud rationale:")
    print(fraud_result.get("fraud_reason", "(no fraud_reason returned)"))

    _subcard("Models Used")
    print(f"NLP model: {used_nlp_model}")
    print(f"Fraud model: {used_fraud_model}")

    print()
    print("END OF RUN")


if __name__ == "__main__":
    asyncio.run(main())

