from __future__ import annotations

import json
import mimetypes
import re
import zipfile
from pathlib import Path
from typing import Any, List
from xml.etree import ElementTree as ET

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field


class DocumentTextInput(BaseModel):
    file_path: str = Field(..., description="Path to source document.")
    detected_file_type: str = Field(..., description="Detected file type/format.")
    extracted_text: str = Field(..., description="Raw extracted text from the file.")


class DocumentTextOutput(BaseModel):
    source_file: str = Field(..., description="Source file path that was processed.")
    detected_file_type: str = Field(..., description="Detected file type/format.")
    normalized_text: str = Field(..., description="Clean text-only output for downstream agents.")
    extraction_notes: List[str] = Field(
        default_factory=list,
        description="Warnings/notes such as OCR fallback or unsupported format details.",
    )
    character_count: int = Field(..., description="Character count of normalized_text.")


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Final fallback for unknown encodings.
    return path.read_bytes().decode("utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> tuple[str, list[str]]:
    notes: list[str] = []
    try:
        from pypdf import PdfReader
    except Exception:
        return "", ["PDF support needs `pypdf` installed."]

    text_parts: list[str] = []
    reader = PdfReader(str(path))
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"[Page {idx}]\n{page_text}")
    if not text_parts:
        notes.append("No text extracted from PDF (possibly scanned/image-only PDF).")
    return "\n\n".join(text_parts), notes


def _read_docx_file(path: Path) -> tuple[str, list[str]]:
    notes: list[str] = []
    paragraphs: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            xml_content = zf.read("word/document.xml")
        root = ET.fromstring(xml_content)
        # WordprocessingML text nodes typically end with '}t'
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                paragraphs.append(node.text)
    except Exception as exc:
        return "", [f"DOCX extraction failed: {exc}"]
    if not paragraphs:
        notes.append("No text nodes found in DOCX file.")
    return "\n".join(paragraphs), notes


def _read_any_file_to_text(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    mime = mimetypes.guess_type(str(path))[0] or "unknown"
    notes: list[str] = []

    if ext in {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".yaml", ".yml", ".log"}:
        text = _read_text_file(path)
        detected = ext.lstrip(".")
    elif ext == ".pdf":
        text, pdf_notes = _read_pdf_file(path)
        notes.extend(pdf_notes)
        detected = "pdf"
    elif ext == ".docx":
        text, docx_notes = _read_docx_file(path)
        notes.extend(docx_notes)
        detected = "docx"
    elif ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        # OCR is intentionally not hard-coupled here to avoid external system deps.
        text = ""
        detected = "image"
        notes.append(
            "Image detected. OCR tool is not configured in this local extractor yet; "
            "add OCR integration before using image-only inputs."
        )
    else:
        text = _read_text_file(path)
        detected = f"unknown ({mime})"
        notes.append("Unknown extension; attempted best-effort text decode.")

    # Normalize whitespace for downstream agents.
    text = re.sub(r"\r\n?", "\n", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {
        "detected_file_type": detected,
        "extracted_text": text,
        "extraction_notes": notes,
    }


def create_document_to_text_agent(model: str) -> LlmAgent:
    instruction = """
You are a medical-document normalization NLP agent.

You will receive:
- file_path
- detected_file_type
- extracted_text

Task:
1) Keep all clinically relevant and factual content exactly as text.
2) Clean formatting noise, remove repeated blank lines, preserve section meaning.
3) Do NOT summarize away facts. Keep medications, dates, diagnoses, dosages, observations.
4) Output plain text in `normalized_text` that downstream agents can parse.

Output must be valid JSON matching the schema.
""".strip()

    return LlmAgent(
        model=model,
        name="nlp_document_to_text_agent",
        description="Converts extracted document content into clean, downstream-ready text.",
        instruction=instruction,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
        input_schema=DocumentTextInput,
        output_schema=DocumentTextOutput,
        output_key="document_text_result",
    )


async def extract_document_to_text(
    *,
    file_path: str,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """
    End-to-end local document-to-text pipeline:
    - Detect file type
    - Extract text with local parsers
    - Use ADK LlmAgent to normalize into clean TXT
    """

    src = Path(file_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Document not found: {src}")
    if src.is_dir():
        raise IsADirectoryError(f"Expected a file, got directory: {src}")

    extracted = _read_any_file_to_text(src)

    input_payload = {
        "file_path": str(src),
        "detected_file_type": extracted["detected_file_type"],
        "extracted_text": extracted["extracted_text"],
    }

    agent = create_document_to_text_agent(model=model)
    session_service = InMemorySessionService()
    app_name = "insurance_doc_to_text_app"
    user_id = "local_user"
    session_id = "doc_to_text_session"
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=json.dumps(input_payload))])
    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None) and part.text.strip():
                    final_text = part.text.strip()
                    break

    if not final_text:
        raise RuntimeError("No final response from document-to-text NLP agent.")

    result = DocumentTextOutput.model_validate_json(final_text).model_dump(exclude_none=True)
    # Preserve local extraction notes from deterministic parser stage.
    parser_notes = extracted.get("extraction_notes", [])
    if parser_notes:
        result["extraction_notes"] = parser_notes + result.get("extraction_notes", [])
    return result


class TobaccoEvidenceInput(BaseModel):
    raw_medical_text: str = Field(..., description="Free-form medical note text to analyze.")


class TobaccoEvidenceOutput(BaseModel):
    tobacco_nicotine_evidence_found: bool = Field(
        ..., description="Whether the text contains evidence of tobacco or nicotine use."
    )
    inferred_smoker_status: bool = Field(
        ..., description="Whether a smoker/nicotine user is likely based on the evidence."
    )
    evidence_snippets: List[str] = Field(
        default_factory=list,
        description="Short evidence snippets (quoted phrases) that justify the boolean. Empty if none found.",
    )
    evidence_summary: str = Field(
        ..., description="A brief 1-2 sentence summary of what the evidence is and why it matters."
    )
    reasoning_summary: str = Field(
        ..., description="A short, human-readable explanation of the decision process."
    )


def create_tobacco_evidence_agent(model: str) -> LlmAgent:
    instruction = """
You are an Insurance Underwriting NLP evidence extraction agent.

Goal: Given a medical note, decide whether there is any evidence of tobacco or nicotine use.

Rules:
1. Scan for nicotine/tobacco indicators such as: cigarettes, smoking, tobacco, nicotine, patches (e.g., Nicotex), gum, vaping, etc.
2. Evidence must be explicitly grounded in the provided text. If there is no explicit evidence, return false.
3. evidence_snippets must be short quoted substrings from the provided text (up to ~12 words each). If none, return [].
4. inferred_smoker_status should be true if tobacco/nicotine evidence is found; else false.

Output rules (strict):
- Return ONLY a single valid JSON object (no markdown, no code fences).
- No trailing commas.
- Use JSON booleans `true`/`false`.
- The JSON must contain ALL keys in the output schema.

JSON keys you must output:
{
  "tobacco_nicotine_evidence_found": boolean,
  "inferred_smoker_status": boolean,
  "evidence_snippets": string[],
  "evidence_summary": string,
  "reasoning_summary": string
}
""".strip()

    return LlmAgent(
        model=model,
        name="nlp_tobacco_evidence_agent",
        description="Extracts tobacco/nicotine evidence from medical text.",
        instruction=instruction,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
        input_schema=TobaccoEvidenceInput,
        output_schema=TobaccoEvidenceOutput,
        output_key="nlp_result",
    )


# Backward-compatible alias used by current main.py
def create_nlp_agent(model: str) -> LlmAgent:
    return create_tobacco_evidence_agent(model)

