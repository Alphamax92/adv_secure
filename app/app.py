import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import spacy

from transformers import (
    pipeline,
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from huggingface_hub import hf_hub_download
from openai import OpenAI

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
set_seed(42)
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if DEVICE == 0 else "cpu"
st.write(f"Using device: {'CUDA (GPU)' if DEVICE == 0 else 'CPU'}")

def ensure_ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")

# ------------------------------------------------------------
# PHI redaction (spaCy with resilient fallback) + regex backup
# ------------------------------------------------------------
@st.cache_resource
def load_phi_model():
    try:
        import spacy  # Import spacy inside the function to fix variable scope issue
        return spacy.load("en_core_web_sm")
    except Exception as e_first:
        st.warning(f"spaCy en_core_web_sm load failed: {e_first}. Attempting download...")
        try:
            import spacy
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e_download:
            st.warning(f"spaCy model download failed: {e_download}. Falling back to blank('en') without NER.")
            import spacy
            return spacy.blank("en")

nlp_phi = load_phi_model()

PHI_REGEX_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4})\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"), "[REDACTED_DATE]"),
    (re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive)\b", re.IGNORECASE), "[REDACTED_ADDRESS]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]

PHI_LABELS = [
    "PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW",
    "LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
]

def sanitize_input(text: str):
    out = text
    redactions = []
    try:
        doc = nlp_phi(text)
        if "ner" in nlp_phi.pipe_names and getattr(doc, "ents", None):
            mutable = list(text)
            for ent in sorted(doc.ents, key=lambda e: e.end_char, reverse=True):
                if ent.label_ in PHI_LABELS:
                    placeholder = f"[REDACTED_{ent.label_}]"
                    mutable[ent.start_char:ent.end_char] = list(placeholder)
                    redactions.append({"entity": ent.text, "label": ent.label_, "placeholder": placeholder})
            out = "".join(mutable)
    except Exception as e:
        st.warning(f"spaCy NER failed during sanitization: {e}. Continuing with regex fallback.")

    for pattern, placeholder in PHI_REGEX_PATTERNS:
        def repl(m):
            redactions.append({"entity": m.group(0), "label": "REGEX", "placeholder": placeholder})
            return placeholder
        out = pattern.sub(repl, out)
    return out, redactions

# ------------------------------------------------------------
# High-risk input safety gate
# ------------------------------------------------------------
HIGH_RISK_PATTERNS = {
    "self_harm": [
        r"\bkill myself\b",
        r"\bcommit (?:suicide|self[-\s]?harm)\b",
        r"\bend my life\b",
        r"\bhow (?:do|to) (?:i )?(?:die|end it)\b",
        r"\b(?:harm|hurt)\s+myself\b",
        r"\boverdos(e|ing)\b",
        r"\bhow (?:much|many)\s+(?:pills?|tablets?|mg|milligrams)\s+(?:to|should i)\s+(?:take|use)\b",
        r"\b(?:knock|pass)\s+(?:me|myself)\s+out\b",
        r"\binduce\s+(?:unconsciousness|black(?:\s|-)?out|faint(?:ing)?)\b",
    ],
    "chemical_misuse": [
        r"\bchloroform\b",
        r"\bhow (?:to|do i) make\b.*\bchloroform\b",
        r"\bether\b.*\bknock (?:someone|me) out\b",
        r"\bwhat chemical\b.*\binduce\s+unconsciousness\b",
    ],
    "violence": [
        r"\bhow (?:to|do i) harm\b",
        r"\bdisable someone\b",
        r"\bmake a weapon\b",
    ]
}

def detect_high_risk_intent(text: str):
    lowered = text.lower()
    hits = []
    for cat, patterns in HIGH_RISK_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lowered, flags=re.IGNORECASE):
                hits.append({"category": cat, "pattern": pat})
    categories = sorted({h["category"] for h in hits})
    return categories, hits

CRISIS_MESSAGE = (
    "I'm really sorry you're feeling this way. I can't help with requests about self-harm, overdose, or causing unconsciousness. "
    "If you're in immediate danger or thinking about harming yourself, please contact your local emergency number now. "
    "If you're in the United States, you can call or text 988 to reach the Suicide & Crisis Lifeline. "
    "If you're outside the U.S., please contact your local emergency services or a trusted crisis hotline in your country. "
    "You're not alone, and reaching out for immediate help is a strong and important step."
)

# ------------------------------------------------------------
# Adversarial detection with calibrated model
# ------------------------------------------------------------
UNCERTAINTY_THRESHOLD_LOW = 0.50
UNCERTAINTY_THRESHOLD_HIGH = 0.80

@st.cache_resource
def load_adv_secure_model():
    repo_id = "alpha-max/adv_secure_v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        try:
            params_path = hf_hub_download(repo_id=repo_id, filename="calibration_params.pt")
            params = torch.load(params_path, map_location=device, weights_only=False)
            optimal_temperature = float(params.get("temperature", 1.0))
            optimal_threshold = float(params.get("threshold", 0.5))
            st.info(f"adv_secure_v2 calibration loaded: T={optimal_temperature:.4f}, thr={optimal_threshold:.4f}")
        except Exception as e_cal:
            st.warning(f"Calibration params not found/failed ({e_cal}). Using defaults T=1.0, thr=0.5")
            optimal_temperature = 1.0
            optimal_threshold = 0.5

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "optimal_temperature": optimal_temperature,
            "optimal_threshold": optimal_threshold,
        }
    except Exception as e:
        st.warning(f"adv_secure_v2 load failed: {e}. Falling back to pipeline classifier.")
        return None

adv_secure = load_adv_secure_model()

def predict_text_with_calibration(text: str, model, tokenizer, device, optimal_temperature: float, optimal_threshold: float, max_length: int = 32):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        raw_logits = outputs.logits

    calibrated_logits = raw_logits / optimal_temperature
    calibrated_probabilities = torch.nn.functional.softmax(calibrated_logits, dim=-1)

    adversarial_prob = calibrated_probabilities[0][1].item()
    normal_prob = calibrated_probabilities[0][0].item()

    predicted_class_idx = 1 if adversarial_prob >= optimal_threshold else 0
    confidence = calibrated_probabilities[0][predicted_class_idx].item()

    if UNCERTAINTY_THRESHOLD_LOW < confidence < UNCERTAINTY_THRESHOLD_HIGH:
        prediction_status = "Uncertain"
        if predicted_class_idx == 1:
            predicted_label_text = "Uncertain (Adversarial bias)"
        else:
            predicted_label_text = "Uncertain (Normal bias)"
    else:
        predicted_label_text = "Adversarial" if predicted_class_idx == 1 else "Normal"
        prediction_status = "Certain"

    return {
        "text": text,
        "classification": predicted_label_text,
        "status": prediction_status,
        "confidence": confidence,
        "probabilities": {"normal": normal_prob, "adversarial": adversarial_prob}
    }

@st.cache_resource
def load_adversarial_detector_fallback():
    try:
        return pipeline("text-classification", model="alpha-max/adv-telemedicine", device=DEVICE)
    except Exception as e:
        st.warning(f"alpha-max/adv-telemedicine failed ({e}). Falling back to sentiment (CPU).")
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

adversarial_detector_fallback = load_adversarial_detector_fallback()

def is_adversarial(text: str):
    cats, _hits = detect_high_risk_intent(text)
    gate_adv = len(cats) > 0

    details = {
        "gate_categories": cats,
        "model_used": None,
        "label": None,
        "confidence": None,
        "probabilities": None,
        "status": None,
    }

    if adv_secure is not None:
        details["model_used"] = "alpha-max/adv_secure_v2"
        pred = predict_text_with_calibration(
            text=text,
            model=adv_secure["model"],
            tokenizer=adv_secure["tokenizer"],
            device=adv_secure["device"],
            optimal_temperature=adv_secure["optimal_temperature"],
            optimal_threshold=adv_secure["optimal_threshold"],
            max_length=32,
        )
        label = pred["classification"]
        conf = float(pred["confidence"])
        details.update({
            "label": label,
            "confidence": conf,
            "probabilities": pred["probabilities"],
            "status": pred["status"],
        })

        model_adv = label.startswith("Adversarial") or label.startswith("Uncertain (Adversarial")
        is_adv = gate_adv or model_adv
        return is_adv, conf, cats, details

    res = adversarial_detector_fallback(text)[0]
    model_name = getattr(getattr(adversarial_detector_fallback, "model", None), "name_or_path", "")
    details["model_used"] = model_name or "fallback_pipeline"
    details["label"] = res.get("label")
    details["confidence"] = float(res.get("score", 0))
    details["status"] = "N/A"

    if model_name == "alpha-max/adv-telemedicine":
        model_adv = res["label"] == "LABEL_1"
    else:
        model_adv = res["label"].upper() == "NEGATIVE" and res["score"] > 0.9

    is_adv = gate_adv or model_adv
    return is_adv, float(res["score"]), cats, details

# ------------------------------------------------------------
# Generation via OpenAI through OpenRouter
# ------------------------------------------------------------
def get_openrouter_api_key():
    """Get OpenRouter API key from environment variables or Streamlit secrets"""
    return "Your open ai api key here"

def generate_response(prompt_text: str) -> str:
    api_key = get_openrouter_api_key()

    if not api_key:
        return "Error: OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable or add it to Streamlit secrets."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": (
            "You are a cautious, non-diagnostic medical assistant. "
            "Provide general, safe health information. Never diagnose or prescribe. "
            "If a request involves self-harm, overdose, inducing unconsciousness, or violence, refuse and provide crisis resources. "
            "Encourage consulting a qualified clinician for concerns."
        )},
        {"role": "user", "content": ensure_ascii(prompt_text)},
    ]

    try:
        chat = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://secure-telemedicine-chatbot.streamlit.app",
                "X-Title": "Secure Telemedicine Chatbot",
            },
            extra_body={},
            messages=messages,
            model="openai/gpt-5-chat",
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            stream=False,
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"Generation error: {e}"

# ------------------------------------------------------------
# Simple LLM-based output verification (replaces complex ensemble)
# ------------------------------------------------------------
def simple_safety_verification(bot_output: str) -> tuple[bool, str]:
    api_key = get_openrouter_api_key()

    if not api_key:
        return False, "UNSAFE - OpenRouter API key not found"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    verification_prompt = f"""You are a medical safety verification system. Your job is to review the chatbot's output and decide if it is safe for the user.

1. Input: This is the chatbot's answer to a medical question:
"{bot_output}"

2. Check the answer for:
   - Harmful or unsafe medical advice (e.g., dangerous dosages, unverified treatments)
   - False or misleading medical claims
   - Encouragement of risky behavior
   - Advice outside general, non-diagnostic guidance
   - Any inclusion of protected health information (PHI)

3. If the answer is SAFE:
   - Reply with: "SAFE"

4. If the answer is UNSAFE:
   - Reply with: "UNSAFE" and a short reason why

Only reply with "SAFE" or "UNSAFE" plus a brief explanation if unsafe."""

    try:
        chat = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://secure-telemedicine-chatbot.streamlit.app",
                "X-Title": "Secure Telemedicine Chatbot",
            },
            extra_body={},
            messages=[{"role": "user", "content": verification_prompt}],
            model="openai/gpt-5-chat",
            temperature=0,
            max_tokens=100,
            stream=False,
        )

        response = chat.choices[0].message.content.strip()

        if response.startswith("SAFE"):
            return True, "SAFE"
        elif response.startswith("UNSAFE"):
            return False, response
        else:
            return False, f"UNSAFE - Unclear verification response: {response}"

    except Exception as e:
        return False, f"UNSAFE - Verification error: {e}"

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = pd.DataFrame(columns=[
        "timestamp","user_input","redacted_input","phi_redactions",
        "safety_input_flag","safety_input_categories",
        "is_adversarial","adversarial_confidence","adversarial_label","adversarial_probs","adversarial_model","adversarial_status",
        "model_response",
        "verification_result","verification_reason",
        "final_status","final_output"
    ])

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Secure Telemedicine Chatbot")
    st.title("Secure Telemedicine Chatbot")
    st.caption("Multi-layer security: Adversarial Detection → PHI Redaction → Safety Gate → OpenAI Generation → Output Verification")

    with st.sidebar:
        st.header("Admin")
        show_dashboard = st.checkbox("Show Admin Dashboard")
        show_checks = st.checkbox("Show Internal Checks")
        auto_disclaimer = st.checkbox("Auto-append disclaimer if missing", value=True)

    if show_dashboard:
        st.subheader("Admin Dashboard")
        st.dataframe(st.session_state.chat_logs)
        total = len(st.session_state.chat_logs)
        if total:
            blocked = (st.session_state.chat_logs["final_status"] == "Blocked").sum()
            st.write(f"Total: {total} | Blocked: {blocked} ({blocked/total*100:.1f}%)")
            counts = st.session_state.chat_logs["final_status"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#66b3ff","#ff9999"])
            ax.axis("equal")
            st.pyplot(fig)
        return

    st.header("Chat")
    user_input = st.text_area("Enter your medical query:", height=120)

    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a query.")
            return

        ts = datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "user_input": user_input,
            "redacted_input": "",
            "phi_redactions": [],
            "safety_input_flag": False,
            "safety_input_categories": [],
            "is_adversarial": False,
            "adversarial_confidence": 0.0,
            "adversarial_label": None,
            "adversarial_probs": None,
            "adversarial_model": None,
            "adversarial_status": None,
            "model_response": "",
            "verification_result": None,
            "verification_reason": "",
            "final_status": "Processed",
            "final_output": ""
        }

        with st.spinner("Processing..."):
            is_adv, adv_conf, adv_cats, adv_details = is_adversarial(user_input)
            entry["is_adversarial"] = is_adv
            entry["adversarial_confidence"] = adv_conf
            entry["adversarial_label"] = adv_details.get("label")
            entry["adversarial_model"] = adv_details.get("model_used")
            entry["adversarial_status"] = adv_details.get("status")
            entry["adversarial_probs"] = adv_details.get("probabilities")
            if show_checks:
                st.info(f"Adversarial: {'YES' if is_adv else 'no'} (conf={adv_conf:.2f})")
                st.json(adv_details)
            if is_adv:
                entry["final_status"] = "Blocked"
                entry["final_output"] = ""
                log_chat(entry)
                st.error("Your request appears unsafe or adversarial. I can't assist with that.")
                return

            redacted, redacts = sanitize_input(user_input)
            entry["redacted_input"] = redacted
            entry["phi_redactions"] = redacts
            if show_checks:
                st.info(f"PHI-redacted: {redacted}")
                if redacts: st.json(redacts)

            cats, hits = detect_high_risk_intent(redacted)
            entry["safety_input_flag"] = len(cats) > 0
            entry["safety_input_categories"] = cats
            if show_checks and hits:
                st.info(f"High-risk input categories: {cats}")
                st.json(hits)
            if cats:
                entry["final_status"] = "Blocked"
                entry["verification_reason"] = f"Blocked: High-risk input ({', '.join(cats)})"
                entry["final_output"] = ""
                log_chat(entry)
                st.error(CRISIS_MESSAGE)
                return

            resp = generate_response(redacted)
            entry["model_response"] = resp
            if show_checks:
                st.info("Model response:")
                st.write(resp)

            is_safe, verification_reason = simple_safety_verification(resp)
            entry["verification_result"] = is_safe
            entry["verification_reason"] = verification_reason

            if show_checks:
                st.info(f"Verification: {verification_reason}")

            final_out = enforce_disclaimer(resp) if auto_disclaimer else resp

            if not is_safe:
                entry["final_status"] = "Blocked"
                entry["final_output"] = ""
                log_chat(entry)
                st.error(f"🚫 {verification_reason}")
                return

            entry["final_status"] = "Allowed"
            entry["final_output"] = final_out
            log_chat(entry)
            st.success("✅ Response:")
            st.write(final_out)

    st.markdown("---")
    st.subheader("Feedback")
    fb = st.text_area("Tell us how we can improve:", height=80)
    if st.button("Submit Feedback"):
        st.success("Thanks for your feedback!")

def log_chat(entry):
    st.session_state.chat_logs = pd.concat([st.session_state.chat_logs, pd.DataFrame([entry])], ignore_index=True)

def enforce_disclaimer(response):
    disclaimer = "Disclaimer: This response is for general information only and should not be considered medical advice."
    if disclaimer not in response:
        response += "\n\n" + disclaimer
    return response

if __name__ == "__main__":
    main()
