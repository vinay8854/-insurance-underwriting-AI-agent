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
from agents.clinical_reasoning_agent import evaluate_clinical_risk, create_clinical_reasoning_agent, cache_store
from agents.policy_generation_agent import create_policy_generation_agent, generate_policy_text_file
from google.adk.models.lite_llm import LiteLlm


APP_NAME = "insurance_adk_demo"
USER_ID = "demo_user_001"
SESSION_ID = "underwriting_session_001"


def _find_json_object(text: str) -> dict[str, Any]:
    """
    Attempts to extract the first JSON object found in the model output.
    ADK output_schema *should* already return pure JSON, but this is a safety net.
    """
    # Pre-processing: Strip markdown and conversational preamble
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json")[1].split("```")[0].strip()
    elif "```" in candidate:
        candidate = candidate.split("```")[1].split("```")[0].strip()

    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No opening brace '{{' found in LLM output. Raw: {text[:200]}...")

    # Robust extraction: find the matching closing brace if possible
    # If missing, we try to return what we have (best effort parsing)
    json_str = candidate[start:]
    
    # Try to clean up trailing text (e.g. "Note: ...")
    last_brace = json_str.rfind("}")
    if last_brace != -1:
        json_str = json_str[:last_brace + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: try to manually close the object if it's just a missing bracket
        try:
            return json.loads(json_str + "}")
        except:
            raise ValueError(f"Failed to parse JSON even with recovery. Raw: {json_str[:200]}...")


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

    gemini_model: str | LiteLlm = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if not gemini_model:
        raise RuntimeError("Missing GEMINI_MODEL. Set it in `.env`.")

    # If the model name looks like a LiteLLM provider (e.g. "groq/", "openai/"), wrap it.
    if isinstance(gemini_model, str) and ("/" in gemini_model or gemini_model.startswith("groq")):
        print(f"Using LiteLLM for model: {gemini_model}")
        gemini_model = LiteLlm(model=gemini_model)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key and not isinstance(gemini_model, LiteLlm):
        raise RuntimeError(
            "Missing Gemini API key and not using LiteLLM. Set `GOOGLE_API_KEY` in a local .env file."
        )
    dummy_payload_path = base_dir / "data" / "dummy_payload.json"
    with dummy_payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Use data/data.txt as the primary source for medical notes if it exists
    data_txt_path = base_dir / "data" / "data.txt"
    if data_txt_path.exists():
        medical_notes = data_txt_path.read_text(encoding="utf-8")
        payload["raw_medical_text"] = medical_notes
        print(f"Loaded detailed clinical notes from {data_txt_path.name}")

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

    # Step E: Clinical Reasoning (New)
    clinical_result = None
    meds = nlp_result.get("medications", [])
    if meds:
        _subcard("Step E: Clinical Reasoning & Pharmacology (ADK + openFDA)")
        # Gathering raw data from APIs
        raw_assessments = await evaluate_clinical_risk(meds, model=gemini_model)
        
        # Now run the LLM agent over that collected data
        clinical_agent = create_clinical_reasoning_agent(gemini_model)
        attempt_session_id = f"{SESSION_ID}_clinical"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=attempt_session_id
        )
        clinical_runner = Runner(agent=clinical_agent, app_name=APP_NAME, session_service=session_service)
        
        clinical_result = await run_llm_agent(
            clinical_agent,
            clinical_runner,
            session_service,
            attempt_session_id,
            input_data={"contexts": raw_assessments},
        )
        print("Clinical logic summary:")
        print(json.dumps(clinical_result, indent=2))
        
        # Cache write-back (demo)
        for idx, assess in enumerate(clinical_result.get("assessments", [])):
            if idx < len(meds):
                cache_store.set(meds[idx], assess)

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

    # --- HUMAN IN THE LOOP (HITL) ---
    final_smoker_status = payload["declared_smoker"]
    if fraud_result.get("fraud_contradiction"):
        _subcard("HUMAN IN THE LOOP: Fraud Resolution Needed!")
        print(f"FRAUD SIGNAL: {fraud_result.get('fraud_reason')}")
        print(f"Evidence found: {fraud_result.get('key_evidence_snippets', [])}")
        print()
        print("ACTION REQUIRED: How do you want to proceed?")
        print("1. Type 'force' to move user to Smoker category and apply fraud load.")
        print("2. Type 'accept' to ignore findings and proceed as Non-Smoker.")
        
        # Wait for terminal input (demo mode)
        user_choice = input("\nDecision [force/accept]: ").strip().lower()
        
        if user_choice == 'force':
            print(">>> UNDERWRITER ACTION: Manually forcing Smoker Status + Fraud Load.")
            fraud_result["fraud_risk_flag"] = True
            final_smoker_status = True
        else:
            print(">>> UNDERWRITER ACTION: Accepting Application Statement (No Fraud Load).")
            fraud_result["fraud_risk_flag"] = False
            final_smoker_status = False

    # Step C: Pricing
    _subcard("Step C: Pricing Load (Pure Python - Aggregate Risk)")
    base_premium = 10000
    
    # Get the risk class from clinical results if available, else default to 'Standard'
    med_risk_class = clinical_result.get("overall_medical_risk_class", "Standard") if clinical_result else "Standard"
    
    pricing = calculate_premium(
        base_premium=base_premium, 
        fraud_risk_flag=fraud_result["fraud_risk_flag"],
        medical_risk_class=med_risk_class
    )
    print(json.dumps(pricing, indent=2))

    # Step D: Compliance
    _subcard("Step D: Compliance Check (Pure Python)")
    compliance = compliance_check(premium_load_factor=pricing["total_load_factor"])
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
    print(f"Medical Risk Category: {med_risk_class}")
    print(f"Applied premium load: {pricing['total_load_factor']}")
    print(f"Final Decision: {pricing.get('underwriting_decision')}")
    print(
        f"Compliant load (<= {compliance['tobacco_load_max_cap']}): {compliance['compliant']}"
    )

    # Step F: Policy Generation (New)
    _subcard("Step F: Policy Issuance & Document Generation")
    policy_input = {
        "applicant_name": payload["name"],
        "underwriting_decision": pricing.get("underwriting_decision"),
        "total_premium": pricing.get("premium_total"),
        "risk_logic": clinical_result.get("clinical_summary") if clinical_result else "Standard evaluation",
        "total_load_factor": pricing.get("total_load_factor")
    }
    
    policy_agent = create_policy_generation_agent(gemini_model)
    attempt_session_id = f"{SESSION_ID}_policy"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=attempt_session_id
    )
    policy_runner = Runner(agent=policy_agent, app_name=APP_NAME, session_service=session_service)
    
    policy_json = await run_llm_agent(
        policy_agent,
        policy_runner,
        session_service,
        attempt_session_id,
        input_data=policy_input,
    )
    
    generated_path = generate_policy_text_file(payload["name"], policy_json)
    print(f"Policy issued successfully!")
    print(f"Document saved to: {generated_path}")
    print()
    print("Issuance Summary Preview:")
    print(policy_json.get("policy_body_text", "").strip()[:300] + "...")

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
    if clinical_result:
        print(f"Clinical Risk: {clinical_result.get('overall_medical_risk_class')}")
        print(f"Risk Logic: {clinical_result.get('clinical_summary')}")

    print()
    print("END OF RUN")


if __name__ == "__main__":
    asyncio.run(main())

