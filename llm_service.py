"""
Student Score Predictor - LLM Service Module
Phase 9: AI / LLM Integration

Responsibilities:
- Manage external LLM configuration via environment variables (LLM_API_KEY, LLM_MODEL)
- Isolate LLM provider logic (Google Gemini via official google-genai SDK)
- Construct secure, structured prompts with strict system boundaries
- Prevent prompt injection and enforce separation of responsibilities:
  * ML Model: Numerical score prediction
  * LLM: Prediction explanation and personalized study recommendations
- Handle API errors, rate limits, timeouts, and missing API keys gracefully
"""

import os
import logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Default model recommendations
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_TIMEOUT_MS = 15000  # 15 seconds timeout


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

    # Determine provider and sensible default model
    if api_key.startswith("gsk_"):
        # Groq key provided
        if not model_name or model_name.startswith("gemini"):
            model_name = DEFAULT_GROQ_MODEL
    else:
        # Default to Gemini
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


def build_study_advice_prompt(student_data: Dict[str, float], predicted_score: float) -> Tuple[str, str]:
    """
    Construct safe, isolated system instructions and user prompt content.
    
    Treats numerical inputs as pure data to prevent prompt injection.
    Reinforces that the ML model produced the numerical prediction.
    """
    system_instruction = (
        "You are an encouraging and practical academic study advisor assisting students in understanding their score predictions.\n"
        "The student's predicted score has ALREADY been calculated by a trained Machine Learning model.\n"
        "Do NOT recalculate, change, or dispute the predicted score.\n\n"
        "Your task:\n"
        "1. In 2-3 sentences, explain the prediction in student-friendly terms based on the provided metrics.\n"
        "2. Provide 2-3 practical, realistic study recommendations to help the student improve or maintain their performance.\n\n"
        "Guidelines:\n"
        "- Base your response ONLY on the provided student metrics (daily study hours, previous score, attendance, assignment score).\n"
        "- Do NOT claim absolute certainty or guarantee future test scores.\n"
        "- Do NOT invent student background details not provided.\n"
        "- Keep the response concise, encouraging, and formatted in clean plain text with simple bullet points."
    )

    user_content = (
        f"Student Academic Metrics:\n"
        f"- Daily Study Hours: {student_data.get('study_hours', 0.0):.1f} hours/day\n"
        f"- Previous Exam Score: {student_data.get('previous_score', 0.0):.1f} / 100\n"
        f"- Attendance Rate: {student_data.get('attendance', 0.0):.1f}%\n"
        f"- Assignment Score: {student_data.get('assignment_score', 0.0):.1f} / 100\n"
        f"- ML Model Predicted Score: {predicted_score:.1f} / 100\n\n"
        f"Please explain this prediction and provide actionable study suggestions."
    )

    return system_instruction, user_content


def _generate_with_groq(api_key: str, model_name: str, system_instruction: str, user_content: str) -> Tuple[bool, Dict[str, Any], int]:
    """
    Invoke Groq OpenAI-compatible chat completions API via requests.
    """
    import requests
    import ssl

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "max_tokens": 600
    }

    try:
        timeout_seconds = DEFAULT_TIMEOUT_MS / 1000.0
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error("Groq returned empty choices list.")
                return False, {
                    "success": False,
                    "error": "AI service returned an empty response. Please try again."
                }, 502

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.error("Groq returned empty text content.")
                return False, {
                    "success": False,
                    "error": "AI service returned an empty response. Please try again."
                }, 502

            return True, {
                "success": True,
                "advice": content,
                "model": model_name
            }, 200

        elif response.status_code in (401, 403):
            logger.error("Groq Authentication Failed (status=%s)", response.status_code)
            return False, {
                "success": False,
                "error": "AI service authentication failed. Please verify the configured API key."
            }, 502

        elif response.status_code == 429:
            logger.warning("Groq Rate Limit Exceeded (status=429)")
            return False, {
                "success": False,
                "error": "AI advice is temporarily unavailable because the AI service is busy. Please try again later."
            }, 429

        else:
            logger.error("Groq API returned error HTTP %s: %s", response.status_code, response.text[:200])
            return False, {
                "success": False,
                "error": "AI advice is currently unavailable. Please try again later."
            }, 502

    except requests.exceptions.Timeout:
        logger.error("Groq API request timed out after %s ms.", DEFAULT_TIMEOUT_MS)
        return False, {
            "success": False,
            "error": "AI advice request timed out. Please try again later."
        }, 504

    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException, ssl.SSLError) as net_err:
        logger.error("Groq connection or SSL error: %s", net_err)
        return False, {
            "success": False,
            "error": "AI advice is currently unavailable due to a network connection issue. Please try again later."
        }, 502


def _generate_with_gemini(api_key: str, model_name: str, system_instruction: str, user_content: str) -> Tuple[bool, Dict[str, Any], int]:
    """
    Invoke Google Gemini API via official google-genai SDK.
    """
    import ssl

    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError, ClientError

        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": DEFAULT_TIMEOUT_MS}
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=600
            )
        )

        advice_text = response.text.strip() if response and response.text else ""
        if not advice_text:
            logger.error("Gemini returned an empty response.")
            return False, {
                "success": False,
                "error": "AI service returned an empty response. Please try again."
            }, 502

        return True, {
            "success": True,
            "advice": advice_text,
            "model": model_name
        }, 200

    except ClientError as client_err:
        err_msg = str(client_err)
        logger.error("Gemini ClientError: %s", err_msg)
        if "API_KEY_INVALID" in err_msg or "400" in err_msg or "INVALID_ARGUMENT" in err_msg:
            return False, {
                "success": False,
                "error": "AI service authentication failed. Please verify the configured API key."
            }, 502
        return False, {
            "success": False,
            "error": "AI advice is currently unavailable. Please try again later."
        }, 502

    except APIError as api_err:
        err_msg = str(api_err)
        status_code = getattr(api_err, "code", 500)
        logger.error("Gemini APIError (code=%s): %s", status_code, err_msg)

        if status_code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            return False, {
                "success": False,
                "error": "AI advice is temporarily unavailable because the AI service is busy. Please try again later."
            }, 429

        if status_code in (401, 403) or "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
            return False, {
                "success": False,
                "error": "AI service authentication failed. Please verify the configured API key."
            }, 502

        return False, {
            "success": False,
            "error": "AI advice is currently unavailable. Please try again later."
        }, 502

    except TimeoutError:
        logger.error("Gemini API request timed out after %s ms.", DEFAULT_TIMEOUT_MS)
        return False, {
            "success": False,
            "error": "AI advice request timed out. Please try again later."
        }, 504

    except (ssl.SSLError, ConnectionError, OSError) as conn_err:
        logger.error("Gemini connection / SSL error: %s", conn_err)
        return False, {
            "success": False,
            "error": "AI advice is currently unavailable due to a network connection issue. Please try again later."
        }, 502

    except Exception as exc:
        err_str = str(exc)
        logger.error("Unexpected error in Gemini service: %s", err_str)
        if "timeout" in err_str.lower():
            return False, {
                "success": False,
                "error": "AI advice request timed out. Please try again later."
            }, 504
        if "ssl" in err_str.lower() or "connection" in err_str.lower() or "eof" in err_str.lower():
            return False, {
                "success": False,
                "error": "AI advice is currently unavailable due to a network connection issue. Please try again later."
            }, 502

        return False, {
            "success": False,
            "error": "AI advice is currently unavailable. Please try again later."
        }, 500


def generate_study_advice(student_data: Dict[str, float], predicted_score: float) -> Tuple[bool, Dict[str, Any], int]:
    """
    Generate personalized study advice using the configured LLM API.
    
    Args:
        student_data: Dictionary containing validated student metrics
        predicted_score: Float value representing the validated ML prediction
        
    Returns:
        Tuple of (success: bool, response_payload: dict, http_status_code: int)
    """
    config = get_llm_config()
    api_key = config["api_key"]
    model_name = config["model"]

    # 1. Verify API Key presence
    if not api_key:
        logger.warning("LLM API request attempted but LLM_API_KEY is not configured.")
        return False, {
            "success": False,
            "error": "AI service is not configured. Please set the LLM_API_KEY environment variable."
        }, 503

    # 2. Build prompts
    system_instruction, user_content = build_study_advice_prompt(student_data, predicted_score)

    # 3. Route to provider based on API key prefix or model name
    if api_key.startswith("gsk_") or "groq" in model_name.lower() or "gpt-oss" in model_name.lower():
        return _generate_with_groq(api_key, model_name, system_instruction, user_content)
    else:
        return _generate_with_gemini(api_key, model_name, system_instruction, user_content)

