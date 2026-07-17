"""
app/config.py — Central configuration: model names, MAX_RETRIES, LLM factory.

Extracted from graph.py so every module can import constants without
importing the entire graph-construction machinery.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Pipeline tuning
# ---------------------------------------------------------------------------
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
HF_BASE_URL    = "https://router.huggingface.co/v1"
HF_MAIN_MODEL  = "Qwen/Qwen3-235B-A22B"
HF_GRAD_MODEL  = "meta-llama/Llama-3.3-70B-Instruct"

GROQ_MAIN_MODEL   = "llama-3.3-70b-versatile"
GROQ_GRADER_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# LLM Factory — tries HF Inference Providers, falls back to Groq
# ---------------------------------------------------------------------------
def build_llms():
    """
    Attempts to build both LLMs using the Hugging Face Inference Providers
    router (OpenAI-compatible, free monthly credits).  If the HF token is
    missing, lacks the required 'Inference Providers' permission, or the
    test call fails for any reason, the function transparently falls back
    to the original Groq-hosted models and prints a clear warning.

    Returns:
        (llm, grader_llm)  — a tuple of two ChatModel instances.
    """
    hf_api_key   = os.environ.get("Hf_token")
    groq_api_key = os.environ.get("groq_api_key")

    # ── Try HF Inference Providers first ────────────────────────────────────
    if hf_api_key:
        try:
            print("[*] Testing HF Inference Providers connection...", end=" ", flush=True)
            test_llm = ChatOpenAI(
                model=HF_MAIN_MODEL,
                base_url=HF_BASE_URL,
                api_key=hf_api_key,
                temperature=0,
                max_tokens=8,          # tiny probe — just validate the token
            )
            test_llm.invoke([HumanMessage(content="hi")])
            print("OK")

            # Probe passed — build both production LLMs
            llm = ChatOpenAI(
                model=HF_MAIN_MODEL,
                base_url=HF_BASE_URL,
                api_key=hf_api_key,
                temperature=0,
            )
            grader_llm = ChatOpenAI(
                model=HF_GRAD_MODEL,
                base_url=HF_BASE_URL,
                api_key=hf_api_key,
                temperature=0,
            )
            print(f"[*] Provider : HF Inference Providers")
            print(f"    Main LLM  : {HF_MAIN_MODEL}")
            print(f"    Grader LLM: {HF_GRAD_MODEL}")
            return llm, grader_llm

        except Exception as e:
            err_msg = str(e)
            if "403" in err_msg:
                reason = (
                    "403 Forbidden — your HF token lacks the "
                    "'Make calls to Inference Providers' permission. "
                    "Enable it at huggingface.co/settings/tokens."
                )
            elif "401" in err_msg:
                reason = "401 Unauthorized — check that 'Hf_token' in .env is correct."
            elif "404" in err_msg:
                reason = f"404 Not Found — model '{HF_MAIN_MODEL}' may not exist on this provider."
            else:
                reason = err_msg[:120]

            print(f"FAILED\n[!] HF Inference Providers error: {reason}")
            print("[*] Falling back to Groq models...")
    else:
        print("[!] 'Hf_token' not found in .env — falling back to Groq.")

    # ── Groq fallback ───────────────────────────────────────────────────────
    if not groq_api_key:
        raise EnvironmentError(
            "Neither 'Hf_token' (HF) nor 'groq_api_key' (Groq) is set. "
            "Please add at least one to your .env file."
        )

    llm = ChatGroq(
        model=GROQ_MAIN_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )
    grader_llm = ChatGroq(
        model=GROQ_GRADER_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )
    print(f"[*] Provider : Groq")
    print(f"    Main LLM  : {GROQ_MAIN_MODEL}")
    print(f"    Grader LLM: {GROQ_GRADER_MODEL}")
    return llm, grader_llm
