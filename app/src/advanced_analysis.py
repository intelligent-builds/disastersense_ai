"""
advanced_analysis.py

Gemini wrapper (local-only, no .env). Improved prompt and fallback people-extraction.

Exposes:
- gemini_analyze_text(message: str, force_no_cache: bool=False) -> dict | None

Returned dict (always this shape when not None):
{
  "is_emergency": bool,
  "disaster_type": "collapse"|"fire"|"flood"|"earthquake"|"medical"|"other",
  "people": int | None,
  "summary": str,
  "urgency": "low"|"medium"|"high"|"critical",
  "raw": str (optional raw text for debugging)
}
"""

import re
import json
import time
import threading
import hashlib
from typing import Optional, Dict, Any

# === Hardcoded API key (you said you prefer this locally) ===
GEMINI_API_KEY = "AIzaSyCLCbrZrfBkmw5OL86EjdfBPuuCQaUIwzk"   # <- replace with your actual key in-file
# Use a model name available in your account (use the alias for latest)
GEMINI_MODEL = "models/gemini-flash-latest"

# Cache & rate-limit settings
GEMINI_CACHE_TTL = 300
GEMINI_RATE_PER_MIN = 40

# Try import SDK
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False

_lock = threading.Lock()
_calls = {"count": 0, "reset_at": time.time() + 60}
_cache: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Utilities
# -------------------------
def _make_key(text: str) -> str:
    norm = re.sub(r'\s+', ' ', (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def _cache_get(k: str) -> Optional[Dict[str, Any]]:
    ent = _cache.get(k)
    if not ent: return None
    if ent["expires_at"] < time.time():
        del _cache[k]
        return None
    return ent["value"]

def _cache_set(k: str, v: Dict[str, Any], ttl: int = GEMINI_CACHE_TTL):
    _cache[k] = {"value": v, "expires_at": time.time() + ttl}

def _can_call() -> bool:
    if not GEMINI_SDK_AVAILABLE or not GEMINI_API_KEY:
        return False
    with _lock:
        now = time.time()
        if now >= _calls["reset_at"]:
            _calls["reset_at"] = now + 60
            _calls["count"] = 0
        if _calls["count"] >= GEMINI_RATE_PER_MIN:
            return False
        _calls["count"] += 1
    return True

# -------------------------
# People extraction fallback
# -------------------------
def _extract_people_from_text(message: str, summary: Optional[str] = None) -> Optional[int]:
    """
    Try to extract an integer # of people from message or summary.
    1) look for explicit patterns like '3 people', '3 trapped', '3 victims'
    2) fallback: largest standalone number in text (reasonable upper bound)
    Returns int or None.
    """
    text = ""
    if message:
        text += " " + message
    if summary:
        text += " " + summary
    text = text or ""
    # 1) explicit phrases
    m = re.search(r'\b(\d{1,4})\s+(?:people|persons|victims|trapped|injured|missing|dead|dead persons)\b', text, flags=re.I)
    if m:
        try:
            v = int(m.group(1))
            # ignore obviously implausible values (>10000)
            if 0 < v <= 10000:
                return v
        except Exception:
            pass
    # 2) patterns like 'three people' -> convert words to digits (handle common small words)
    word_map = {
        'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15
    }
    m2 = re.search(r'\b(' + '|'.join(word_map.keys()) + r')\s+(?:people|persons|victims|trapped|injured)\b', text, flags=re.I)
    if m2:
        w = m2.group(1).lower()
        return word_map.get(w)
    # 3) any standalone numbers, take the largest plausible one
    nums = [int(n) for n in re.findall(r'\b(\d{1,4})\b', text)]
    if nums:
        cand = max(nums)
        if 0 < cand <= 10000:
            return cand
    return None

# -------------------------
# JSON parsing helper
# -------------------------
def _parse_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    # find first {...} block
    m = re.search(r'(\{.*\})', raw, flags=re.S)
    if not m:
        return None
    candidate = m.group(1)
    try:
        return json.loads(candidate)
    except Exception:
        # try removing trailing commas
        try:
            cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
            return json.loads(cleaned)
        except Exception:
            return None

# -------------------------
# Main function
# -------------------------
def gemini_analyze_text(message: str, force_no_cache: bool = False) -> Optional[Dict[str, Any]]:
    """
    Analyze `message` and return normalized dict:
      { is_emergency, disaster_type, people, summary, urgency, raw }
    Returns None if Gemini unavailable or call fails.
    """
    if not message or not isinstance(message, str):
        return None

    key = _make_key(message)
    if not force_no_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    if not _can_call():
        # cannot call Gemini now (no SDK/key or rate-limited)
        return None

    # Strong JSON-only prompt. IMPORTANT: ask Gemini to output exact JSON schema.
    prompt = f"""
You are an emergency analysis assistant. Output EXACTLY one JSON object and nothing else.
The JSON MUST contain the following fields:
- is_emergency: a boolean (true or false)
- disaster_type: one of [collapse, fire, flood, earthquake, medical, other]
- people: an integer number of people affected if mentioned, otherwise null
- summary: a single concise sentence summarizing key facts (one line)
- urgency: one of [low, medium, high, critical]

Analyze the message below and output the JSON only.

Message:
\"\"\"{message}\"\"\"
"""

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        # Get raw text
        raw_text = getattr(response, "text", None) or str(response)
        parsed = _parse_json_from_text(raw_text)
        normalized = None
        if parsed and isinstance(parsed, dict):
            # Normalize keys and types
            is_em = parsed.get("is_emergency", parsed.get("emergency", False))
            try:
                is_em_bool = bool(is_em)
            except Exception:
                is_em_bool = True if str(is_em).lower() in ("true","1","yes") else False

            dtype = parsed.get("disaster_type") or parsed.get("type") or "other"
            dtype = str(dtype).lower() if dtype else "other"
            if dtype not in ("collapse","fire","flood","earthquake","medical","other"):
                # map common synonyms
                if "collaps" in dtype: dtype = "collapse"
                elif "fire" in dtype: dtype = "fire"
                elif "flood" in dtype: dtype = "flood"
                elif "earth" in dtype: dtype = "earthquake"
                else: dtype = "other"

            # people - try convert safely to int, allow None
            people_val = parsed.get("people", None)
            people_int = None
            if people_val is not None:
                try:
                    if isinstance(people_val, (int, float)):
                        people_int = int(people_val)
                    else:
                        s = str(people_val).strip()
                        if s.lower() in ("null","none","n/a","na","-"):
                            people_int = None
                        else:
                            # extract digits
                            m = re.search(r'(\d{1,5})', s)
                            if m:
                                people_int = int(m.group(1))
                            else:
                                people_int = None
                except Exception:
                    people_int = None

            summary = str(parsed.get("summary") or "").strip()
            urgency = str(parsed.get("urgency") or "medium").lower()
            if urgency not in ("low","medium","high","critical"):
                # normalize common words
                if "crit" in urgency: urgency = "critical"
                elif "high" in urgency: urgency = "high"
                elif "low" in urgency: urgency = "low"
                else: urgency = "medium"

            normalized = {
                "is_emergency": bool(is_em_bool),
                "disaster_type": dtype,
                "people": people_int,
                "summary": summary,
                "urgency": urgency,
                "raw": raw_text
            }

        # If parsing failed or people missing, apply fallback extraction and normalization
        if not normalized:
            # we didn't get clean JSON — still cache a minimal result to avoid loops
            # but return None to let caller use local model if desired
            return None

        # people fallback: if None, try extract from message/summary
        if normalized.get("people") is None:
            fallback_count = _extract_people_from_text(message, normalized.get("summary"))
            if fallback_count is not None:
                normalized["people"] = int(fallback_count)

        # cache and return
        _cache_set(key, normalized)
        return normalized

    except Exception:
        # on any SDK / network error, return None (caller will fallback)
        return None


# Quick demo when run directly (won't be executed when imported)
if __name__ == "__main__":
    test = "Building has collapsed near MG Road, Bangalore. 3 people trapped!"
    print("Calling gemini_analyze_text on test:", test)
    r = gemini_analyze_text(test, force_no_cache=True)
    print("Result:", r)
