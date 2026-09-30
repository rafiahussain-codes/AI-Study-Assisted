"""
AI Study Assistant - LLM Service Module
Manages external LLM integrations (Google Gemini and Groq)
Supports multi-turn conversational chat completions with error recovery and failover.
"""

import os
import logging
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_MS = 25000  # 25 seconds


def get_llm_config() -> Dict[str, str]:
    """
    Retrieve LLM configuration from environment variables.
    Supports Google Gemini (AIza...) and Groq (gsk_...) providers.
    """
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or ""
    ).strip()
    model_name = (os.environ.get("LLM_MODEL") or "").strip()

    # Determine provider and default model
    if api_key.startswith("gsk_"):
        if not model_name or model_name.startswith("gemini"):
            model_name = DEFAULT_GROQ_MODEL
    else:
        if not model_name:
            model_name = DEFAULT_GEMINI_MODEL

    return {
        "api_key": api_key,
        "model": model_name
    }


def is_llm_configured() -> bool:
    """
    Check if a valid LLM API key has been provided in the environment.
    """
    config = get_llm_config()
    return bool(config["api_key"])


def _chat_with_groq(
    api_key: str,
    model_name: str,
    system_instruction: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1500
) -> Tuple[bool, Dict[str, Any], int]:
    """
    Execute multi-turn conversational chat completion via Groq API.
    """
    import requests
    import ssl

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    groq_messages = [{"role": "system", "content": system_instruction}]
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "assistant"
        groq_messages.append({"role": role, "content": msg.get("content", "")})

    payload = {
        "model": model_name,
        "messages": groq_messages,
        "temperature": 0.5,
        "max_tokens": max_tokens
    }

    try:
        timeout_seconds = max(DEFAULT_TIMEOUT_MS / 1000.0, 20.0)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

        # Handle decommissioned or deprecated model names by falling back to standard Groq model
        if response.status_code in (400, 404) and (
            "decommissioned" in response.text.lower()
            or "not_found" in response.text.lower()
            or "model" in response.text.lower()
        ):
            fallback_model = "llama-3.3-70b-versatile"
            if model_name != fallback_model:
                logger.warning("Groq model '%s' failed. Retrying with '%s'.", model_name, fallback_model)
                payload["model"] = fallback_model
                response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
                if response.status_code == 200:
                    model_name = fallback_model

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return False, {"success": False, "error": "AI service returned an empty choices list."}, 502
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                return False, {"success": False, "error": "AI service returned empty content."}, 502
            return True, {
                "success": True,
                "reply": content,
                "model": model_name
            }, 200

        elif response.status_code in (401, 403):
            return False, {"success": False, "error": "AI service authentication failed. Please verify LLM_API_KEY."}, 502
        elif response.status_code == 429:
            return False, {"success": False, "error": "AI service rate limit reached. Please wait a moment and try again."}, 429
        else:
            logger.error("Groq chat error %s: %s", response.status_code, response.text[:200])
            return False, {"success": False, "error": f"AI service error: {response.text[:120]}"}, 502

    except requests.exceptions.Timeout:
        return False, {"success": False, "error": "AI chat request timed out. Please try again."}, 504
    except (requests.exceptions.RequestException, ssl.SSLError) as exc:
        logger.error("Groq chat connection error: %s", exc)
        return False, {"success": False, "error": "Network error connecting to AI service."}, 502


def _chat_with_gemini(
    api_key: str,
    model_name: str,
    system_instruction: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1500
) -> Tuple[bool, Dict[str, Any], int]:
    """
    Execute multi-turn conversational chat completion via Google Gemini SDK.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": DEFAULT_TIMEOUT_MS}
        )

        formatted_contents = []
        for m in messages:
            role = "user" if m.get("role") == "user" else "model"
            formatted_contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=m.get("content", ""))]
            ))

        response = client.models.generate_content(
            model=model_name,
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5,
                max_output_tokens=max_tokens
            )
        )

        reply_text = response.text.strip() if response and response.text else ""
        if not reply_text:
            return False, {"success": False, "error": "AI service returned an empty response."}, 502

        return True, {
            "success": True,
            "reply": reply_text,
            "model": model_name
        }, 200

    except Exception as exc:
        err_str = str(exc)
        logger.error("Gemini chat error: %s", err_str)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return False, {"success": False, "error": "Gemini rate limit exceeded. Please wait a moment."}, 429
        if "API_KEY_INVALID" in err_str or "401" in err_str:
            return False, {"success": False, "error": "Gemini API key is invalid."}, 502
        if "timeout" in err_str.lower():
            return False, {"success": False, "error": "Gemini request timed out."}, 504
        return False, {"success": False, "error": f"AI service error: {err_str[:120]}"}, 500


def generate_chat_completion(
    messages: List[Dict[str, str]],
    system_instruction: str,
    max_tokens: int = 1500
) -> Tuple[bool, Dict[str, Any], int]:
    """
    Execute conversational multi-turn chat completion using the configured LLM provider.
    """
    config = get_llm_config()
    api_key = config["api_key"]
    model_name = config["model"]

    if not api_key:
        return False, {
            "success": False,
            "error": "AI service is not configured. Please set LLM_API_KEY in your .env file."
        }, 503

    if api_key.startswith("gsk_") or "groq" in model_name.lower() or "gpt-oss" in model_name.lower():
        return _chat_with_groq(api_key, model_name, system_instruction, messages, max_tokens=max_tokens)
    else:
        return _chat_with_gemini(api_key, model_name, system_instruction, messages, max_tokens=max_tokens)
