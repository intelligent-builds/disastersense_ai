# run.py
# Full application entrypoint for Disaster Response project.
# - Integrates Gemini (advanced_analysis) if available
# - Integrates OpenRouteService (ORS) free ETA via matrix endpoint
# - Adds in-memory cache for ORS results, improved formatting, and stores numeric ORS fields
# Credentials are loaded from environment variables; never commit API keys.

import json
import plotly
import pandas as pd
from datetime import datetime
import os
import sys

# ensure project root is on path so "src" imports work when running app/run.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load simple KEY=value entries from the project-root .env without overriding
# real environment variables. This avoids requiring python-dotenv.
def _load_project_env(path):
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('\"').strip("'")
                if key and key.replace("_", "").isalnum():
                    os.environ.setdefault(key, value)
    except OSError:
        pass

_load_project_env(os.path.join(PROJECT_ROOT, ".env"))

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

# extra imports used by the /reports pipeline
import sqlite3
import requests
import re
import math
from collections import deque
import numpy as np
import time

# helper modules (make sure you created these under src/)
from src.preprocess import clean_text
from src.predictors.earthquake import predict_quake
from src.predictors.flood import predict_flood
from src.predictors.tsunami import predict_tsunami
from src.alerts import send_email

# NEW: import Gemini wrapper (your advanced_analysis module)
# The advanced_analysis.py you added should expose gemini_analyze_text(message) -> dict|None
try:
    from src.advanced_analysis import gemini_analyze_text
except Exception:
    # fallback to a stub that returns None (Gemini unavailable)
    def gemini_analyze_text(msg):
        return None

app = Flask(__name__)

def tokenize(text):
    """Tokenize + lemmatize text to match training preprocessing."""
    tokens = word_tokenize(text or "")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data (keep original DB path for SQLAlchemy)
engine = create_engine('sqlite:///../data/DisasterResponse.db')
try:
    df = pd.read_sql_table('Disasters', engine)
except Exception:
    # If the Disasters table / DB isn't present at dev time, create a small empty frame to avoid crashes.
    df = pd.DataFrame(columns=["id", "message", "genre"])

# try to derive category names from the dataframe columns
CATEGORY_COL_START = 4
category_names = list(df.columns[CATEGORY_COL_START:]) if df.shape[1] > CATEGORY_COL_START else []

# load model (keep original path)
MODEL_PATH = "../models/classifier.pkl"
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    app.logger.warning(f"Could not load model at {MODEL_PATH}; continuing without model.")
    model = None


# ----------------------------------------
# Expanded keyword_map (lots of disaster buzzwords)
# ----------------------------------------
keyword_map = {
    "related": [
        "disaster", "emergency", "crisis", "incident", "situation",
        "problem", "help", "urgent", "support", "danger", "mayday", "sos",
        "distress", "alert", "trouble"
    ],
    "request": [
        "request", "please", "need", "require", "asking", "help me",
        "looking for", "appeal", "begging", "pls", "plz", "requesting",
        "need water", "need food", "need shelter", "need help"
    ],
    "offer": [
        "offer", "provide", "donate", "volunteer", "support with", "can give",
        "we will send", "we are offering", "available", "can help", "i can help"
    ],
    "aid_related": [
        "aid", "relief", "assistance", "donation", "charity", "ngo", "supplies",
        "emergency aid", "rescue", "humanitarian", "relief camp", "aid station"
    ],
    "medical_help": [
        "medicine", "medical", "doctor", "hospital", "clinic", "treatment",
        "injury", "injured", "sick", "disease", "ill", "pain", "wound",
        "ambulance", "surgery", "paramedic", "emergency room", "bleeding",
        "fracture", "burns", "serious injury", "need medical", "need doctor"
    ],
    "search_and_rescue": [
        "trapped", "trapped people", "search and rescue", "rescue team", "buried",
        "under rubble", "stuck", "buried alive"
    ],
    "fire": [
        "fire", "burning", "flames", "smoke", "on fire", "blaze"
    ],
    "shelter": [
        "shelter", "camp", "where is the camp", "evacuation centre", "evacuee", "relief camp"
    ],
    "water": [
        "water", "no water", "need water", "drinking water", "water supply"
    ],
    "food": [
        "food", "no food", "starving", "need food", "hunger"
    ],
    "buildings": [
        "collapsed", "collapse", "building collapsed", "rubble", "structure collapse"
    ],
    "death": [
        "dead", "died", "death", "fatal", "killed"
    ],
    "direct_report": [
        "report", "reported", "complain", "told", "said", "direct report", "alert"
    ],
}

# Alerts: send email and append audit log
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
ALERT_LOG_PATH = os.path.join(LOG_DIR, 'alerts.log')

# ---------------------------
# CitizenSOS report processing helpers (location, scoring, helpers)
# ---------------------------

# Path to the existing SQLite DB used by project (keeps everything in one file)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "DisasterResponse.db"))

# Small emergency keyword list (quick-start)
EMERGENCY_KEYWORDS = [
    "collapse","collapsed","trapped","help","urgent","immediate","injured","bleeding",
    "drowning","fire","burning","explosion","attack","shot","kill","killed","dead",
    "fallen","buried","buried alive","stuck","panic","evacuate","rescue",
    "critical","ambulance","hospital","medical", "mayday", "sos"
]

def heuristic_emergency_score(text):
    """Return a heuristic urgency score between 0.0 and 1.0"""
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    # keyword hits
    for kw in EMERGENCY_KEYWORDS:
        if kw in t:
            score += 0.18
    # number-of-people mention increases urgency (small boost)
    nums = re.findall(r'\b(\d{1,4})\b', t)
    if nums:
        try:
            n = max(int(x) for x in nums)
            score += min(0.25, 0.03 * n)
        except Exception:
            pass
    # exclamation marks and 'help!' increases urgency
    score += 0.02 * t.count('!')
    # words like 'please hurry', 'please help' boost
    if "please hurry" in t or "please help" in t:
        score += 0.1
    # clamp
    score = max(0.0, min(1.0, score))
    return round(score, 3)

# ---- REPLACED: improved place extraction + nominatim geocoding ----

PLACE_STOPWORDS = set([
    "building","house","home","here","nearby","near","site","place","area","location",
    "street","road","way","corner","block","shop","office","people","person","trapped","collapsed","collapse","injured","dead","help"
])
GENERIC_TOKENS = set(["building","people","trapped","collapsed","collapse","injured","dead","help","here","nearby","home","where","what","who","when","why","how"])
QUESTION_WORDS = set(["where","what","who","when","why","how"])

def extract_place_candidates(text):
    text = (text or "").strip()
    if not text:
        return []

    patterns = [r'near\s+([A-Za-z0-9\s\-\_\,]+)',
                r'at\s+([A-Za-z0-9\s\-\_\,]+)',
                r'in\s+([A-Za-z0-9\s\-\_\,]+)']
    candidates = []
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            m = m.strip().strip('.,;:')
            if len(m) > 2 and not re.match(r'^(me|home|here|nearby)$', m.strip(), re.I):
                candidates.append(m)

    comma_parts = [part.strip() for part in re.split(r'[,\;\-\\/]', text) if len(part.strip())>3]
    candidates.extend(comma_parts)

    caps = re.findall(r'([A-Z][a-z]{1,}\s(?:[A-Z][a-z]{1,}\s?){0,2})', text)
    candidates.extend([c.strip() for c in caps if len(c.strip())>2])

    words = re.findall(r'\w+', text)
    if len(words) >= 2:
        for n in range(2, min(6, len(words)+1)):
            candidates.append(" ".join(words[-n:]))

    seen = set(); out = []
    for c in candidates:
        key = c.lower().strip()
        tokens = [w for w in re.findall(r'\w+', key)]
        if len(tokens) == 0:
            continue
        if all(t in PLACE_STOPWORDS for t in tokens):
            continue
        if key not in seen:
            seen.add(key); out.append(c)
    return out

def geocode_place_nominatim(place, countrycodes=None, timeout=8):
    if not place:
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': place, 'format': 'json', 'limit': 1}
        if countrycodes:
            params['countrycodes'] = countrycodes
        headers = {'User-Agent': 'CitizenSOS/1.0 (disaster-project)'}
        try:
            app.logger.debug(f"Geocoding: {place}  (countrycodes={countrycodes})")
        except Exception:
            print(f"[geocode] Geocoding: {place}  (countrycodes={countrycodes})")
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            try:
                app.logger.warning(f"Nominatim status {r.status_code} for query: {place}")
            except Exception:
                print(f"[geocode] Nominatim status {r.status_code} for query: {place}")
            return None
        j = r.json()
        if j:
            lat = float(j[0]['lat']); lon = float(j[0]['lon'])
            display = j[0].get('display_name')
            try:
                app.logger.debug(f"Geocode success for '{place}': {lat},{lon} -> {display}")
            except Exception:
                print(f"[geocode] Geocode success for '{place}': {lat},{lon} -> {display}")
            return lat, lon, display
        try:
            app.logger.debug(f"Nominatim returned no results for: {place}")
        except Exception:
            print(f"[geocode] Nominatim returned no results for: {place}")
    except Exception as e:
        try:
            app.logger.warning(f"Geocode error for '{place}': {e}")
        except Exception:
            print(f"[geocode] Geocode error for '{place}': {e}")
    return None

# ---- END replaced helpers ----

def haversine(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon"""
    if None in (lat1,lon1,lat2,lon2):
        return float('inf')
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def find_nearby_helpers(resolved_lat, resolved_lon, max_results=5, max_radius_km=15):
    """Query helpers table and return nearby active helpers sorted by distance"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, name, email, phone, lat, lon, radius_km, active FROM helpers WHERE active=1")
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()
    helpers = []
    for r in rows:
        try:
            hid, name, email, phone, hlat, hlon, hradius, active = r
        except Exception:
            continue
        if hlat is None or hlon is None:
            continue
        dist = haversine(resolved_lat, resolved_lon, hlat, hlon)
        allowed = hradius if (hradius is not None and hradius>0) else max_radius_km
        if dist <= allowed:
            helpers.append({'id': hid, 'name': name, 'email': email, 'phone': phone, 'lat': hlat, 'lon': hlon, 'dist_km': round(dist,2)})
    helpers = sorted(helpers, key=lambda x: x['dist_km'])
    return helpers[:max_results]

def get_eta_approx_km_to_minutes(dist_km, avg_speed_kmph=35):
    """Rough ETA estimate in minutes from distance in km and average speed."""
    try:
        mins = max(1, int((dist_km / float(avg_speed_kmph)) * 60))
        return f"{mins} mins"
    except Exception:
        return "unknown"

# ---------------------------
# OpenRouteService integration (FREE)
# ---------------------------
# Optional OpenRouteService token from environment
ORS_API_KEY = os.getenv("ORS_API_KEY", "")

def get_eta_openrouteservice(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Returns ETA in minutes and distance in km using OpenRouteService (free).
    Uses the Matrix endpoint (driving-car). lon/lat ordering required by ORS.
    Returns dict {distance_km, duration_min, duration_text, duration_s} or None on failure.
    """
    if not ORS_API_KEY:
        return None
    try:
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json"
        }
        # ORS expects [lon, lat]
        payload = {
            "locations": [
                [float(origin_lon), float(origin_lat)],
                [float(dest_lon), float(dest_lat)]
            ],
            "metrics": ["distance", "duration"]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        if r.status_code != 200:
            try:
                app.logger.warning(f"ORS matrix status {r.status_code}: {r.text[:200]}")
            except Exception:
                print(f"[ORS] status {r.status_code}: {r.text[:200]}")
            return None
        j = r.json()
        # debug preview (short)
        try:
            app.logger.debug("ORS response preview: %s", json.dumps(j)[:800])
        except Exception:
            print("[ORS] response preview:", str(j)[:800])

        # durations[0][1] is seconds from origin 0 to dest 1
        durations = j.get("durations")
        distances = j.get("distances")
        if not durations or not distances:
            return None
        duration_s = durations[0][1]
        distance_m = distances[0][1]
        if duration_s is None:
            return None
        return {
            "distance_km": (distance_m / 1000.0) if distance_m is not None else None,
            "duration_min": round(duration_s / 60.0, 1),
            "duration_text": f"{max(1, int(round(duration_s/60.0)))} mins",
            "duration_s": duration_s
        }
    except Exception as e:
        try:
            app.logger.warning(f"ORS ETA error: {e}")
        except Exception:
            print(f"[ORS] ETA error: {e}")
        return None

# ---------------------------
# Simple in-memory ETA cache (TTL)
# ---------------------------
_ETA_CACHE = {}
_ETA_CACHE_TTL = 60  # seconds

def _cache_key_for_eta(o_lat, o_lon, d_lat, d_lon, mode='driving'):
    return f"{float(o_lat):.6f},{float(o_lon):.6f}->{float(d_lat):.6f},{float(d_lon):.6f}|{mode}"

def get_eta_openrouteservice_cached(o_lat, o_lon, d_lat, d_lon, mode='driving'):
    key = _cache_key_for_eta(o_lat, o_lon, d_lat, d_lon, mode)
    now = time.time()
    if key in _ETA_CACHE:
        ts, val = _ETA_CACHE[key]
        if now - ts < _ETA_CACHE_TTL:
            return val
    res = get_eta_openrouteservice(o_lat, o_lon, d_lat, d_lon)
    _ETA_CACHE[key] = (now, res)
    return res

def format_duration_text(duration_text=None, duration_s=None):
    """
    Make short trips nicer:
    - if duration_s < 60 -> "<1 min"
    - else show "N min(s)" (singular/plural normalized)
    """
    try:
        if duration_s is not None:
            if duration_s < 60:
                return "<1 min"
            mins = int(round(duration_s/60.0))
            return f"{mins} min" if mins == 1 else f"{mins} mins"
        if duration_text:
            t = duration_text.strip()
            # normalize common patterns
            if t.startswith("1 ") and "mins" in t:
                return t.replace("mins", "min")
            return t
    except Exception:
        pass
    return duration_text or "unknown"

def notify_helper_via_email(helper, report, eta_text=None):
    """Notify helper via email using src.alerts.send_email. Uses provided eta_text if present."""
    subj = f"URGENT: Nearby emergency — {report.get('short_msg','')[:80]}"
    maps_link = ""
    if report.get('resolved_lat') and report.get('resolved_lon') and helper.get('lat') and helper.get('lon'):
        maps_link = f"https://www.google.com/maps/search/?api=1&query={report['resolved_lat']},{report['resolved_lon']}"
    # prefer provided eta_text; otherwise compute simple ETA
    eta_display = eta_text
    if not eta_display:
        try:
            if report.get('resolved_lat') and report.get('resolved_lon') and helper.get('lat') and helper.get('lon'):
                dist_km = haversine(report['resolved_lat'], report['resolved_lon'], helper.get('lat'), helper.get('lon'))
                eta_display = get_eta_approx_km_to_minutes(dist_km)
        except Exception:
            eta_display = "unknown"
    body = (
        "Hello {},\n\n"
        "A nearby emergency has been reported.\n\n"
        "Message: {}\n"
        "Urgency: {}\n"
        "Location: {}\n"
        "Map: {}\n"
        "Estimated arrival time: {}\n\n"
        "If you can help, please reply to this email or follow local procedures.\n\n"
        "-- CitizenSOS"
    ).format(helper.get('name') or "Volunteer", report.get('message') or "", report.get('urgency'), report.get('location_text') or 'not available', maps_link, eta_display)
    try:
        res = send_email(to_email=helper.get('email'), subject=subj, body=body)
    except Exception as e:
        res = {"status":"error","detail":str(e)}
    return res

# ---------------------------
# Utility: ensure DB columns exist (non-destructive if already present)
# ---------------------------

def ensure_reports_column(column_name, column_type='TEXT'):
    """Add a column to reports table if it doesn't exist. Safe to call repeatedly."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(reports)")
        cols = [r[1] for r in cur.fetchall()]
        if column_name not in cols:
            cur.execute(f"ALTER TABLE reports ADD COLUMN {column_name} {column_type}")
            conn.commit()
        conn.close()
    except Exception:
        try:
            app.logger.warning(f"Could not ensure column {column_name} on reports table")
        except Exception:
            print(f"[db] Could not ensure column {column_name} on reports table")

# ensure ml_info and assigned_helpers columns exist (if the project's DB schema is older)
ensure_reports_column('ml_info', 'TEXT')
ensure_reports_column('assigned_helpers', 'TEXT')

# ---------------------------
# Fusion helpers (NEW)
# ---------------------------
def _gemini_urgency_to_conf(urgency_str):
    """Map Gemini urgency label to heuristic numeric confidence (0..1)."""
    if not urgency_str:
        return 0.5
    s = str(urgency_str).lower()
    if "critical" in s:
        return 0.95
    if "high" in s:
        return 0.85
    if "medium" in s:
        return 0.6
    if "low" in s:
        return 0.35
    return 0.5

def fuse_with_gemini(local_label, local_prob, keyword_score, gemini_result, weights=None):
    """
    Combine local model top prediction (label/prob), keyword_score, and gemini_result.
    Returns dict: {label, confidence, fused_scores}
    """
    if weights is None:
        weights = {"local": 0.6, "gemini": 0.35, "keyword": 0.05}
    fused_scores = {}
    # baseline: local
    fused_scores[local_label] = weights["local"] * float(local_prob) + weights["keyword"] * float(keyword_score)
    # gemini contribution
    if gemini_result:
        g_label = gemini_result.get("disaster_type", "other")
        g_conf = _gemini_urgency_to_conf(gemini_result.get("urgency"))
        fused_scores.setdefault(g_label, 0.0)
        fused_scores[g_label] += weights["gemini"] * g_conf + weights["keyword"] * float(keyword_score)
    # normalize
    total = sum(fused_scores.values()) or 1.0
    for k in list(fused_scores.keys()):
        fused_scores[k] = round(float(fused_scores[k]) / total, 4)
    top_label, top_score = max(fused_scores.items(), key=lambda x: x[1])
    return {"label": top_label, "confidence": float(top_score), "fused_scores": fused_scores}

# ---------------------------
# Flask routes (index and go remain the same, with minor enhancements)
# ---------------------------

@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    try:
        genre_counts = df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)
    except Exception:
        genre_names = []
        genre_counts = []

    try:
        cats = df[df.columns[5:]] if df.shape[1] > 5 else pd.DataFrame()
        cats_counts = cats.mean()*cats.shape[0] if not cats.empty else pd.Series()
        cats_names = list(cats_counts.index) if not cats_counts.empty else []
        nlarge_counts = cats_counts.nlargest(5) if not cats_counts.empty else pd.Series()
        nlarge_names = list(nlarge_counts.index) if not nlarge_counts.empty else []
    except Exception:
        cats = pd.DataFrame()
        cats_counts = pd.Series()
        cats_names = []
        nlarge_counts = pd.Series()
        nlarge_names = []

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=nlarge_names,
                    y=nlarge_counts
                )
            ],
            'layout': {
                'title': 'Top message categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': [
                Bar(
                    x=cats_names,
                    y=cats_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '')
    # Prefer to use same preprocessing as training
    try:
        cleaned_query = " ".join(tokenize(query))
    except Exception:
        cleaned_query = clean_text(query)

    # Prediction: handle predict_proba vs predict for different pipeline types
    classification_labels = None
    try:
        if model is not None and hasattr(model, "predict_proba"):
            raw = model.predict_proba([cleaned_query])
            probs = []
            if isinstance(raw, list):
                for elt in raw:
                    try:
                        val = float(elt[0])
                    except Exception:
                        val = float(elt)
                    probs.append(val)
            else:
                try:
                    probs = list(raw[0])
                except Exception:
                    probs = list(raw)
            threshold = 0.5
            classification_labels = [1 if float(p) >= threshold else 0 for p in probs]
        elif model is not None:
            classification_labels = model.predict([cleaned_query])[0]
        else:
            classification_labels = [0] * len(category_names)
    except Exception:
        try:
            classification_labels = model.predict([cleaned_query])[0] if model is not None else [0] * len(category_names)
        except Exception:
            classification_labels = [0] * len(category_names)

    # Map model outputs to category names robustly
    if len(classification_labels) != len(category_names):
        try:
            classification_labels = list(classification_labels)
        except Exception:
            classification_labels = [0] * len(category_names)
    if len(classification_labels) < len(category_names):
        classification_labels = list(classification_labels) + [0] * (len(category_names) - len(classification_labels))
    elif len(classification_labels) > len(category_names):
        classification_labels = list(classification_labels)[:len(category_names)]

    classification_results = dict(zip(category_names, classification_labels))

    # --- keyword fallback: ensure explicit textual mentions are captured ---
    kw_text = cleaned_query.lower()
    for cat, keys in keyword_map.items():
        for k in keys:
            if k in kw_text:
                if cat in classification_results:
                    classification_results[cat] = 1
                break  # break synonyms loop for this category

    # render go.html with results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# ---------------------------
# Prediction endpoints (earthquake / flood / tsunami) remain unchanged
# ---------------------------

@app.route('/predict/earthquake', methods=['POST'])
def api_predict_quake():
    payload = request.get_json() or {}
    try:
        lat = float(payload.get('lat', 0.0))
        lon = float(payload.get('lon', 0.0))
        depth = float(payload.get('depth', 10.0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid numeric inputs"}), 400

    date = payload.get('date')
    res = predict_quake(lat=lat, lon=lon, date=date, depth=depth)
    return jsonify(res)

@app.route('/predict/flood', methods=['POST'])
def api_predict_flood():
    payload = request.get_json() or {}
    try:
        rainfall = float(payload.get('rainfall_mm', 0.0))
        dam_capacity = float(payload.get('dam_capacity_percent', 50.0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid numeric inputs"}), 400

    res = predict_flood(rainfall_mm=rainfall, dam_capacity_percent=dam_capacity)
    return jsonify(res)

@app.route('/predict/tsunami', methods=['POST'])
def api_predict_tsunami():
    payload = request.get_json() or {}
    try:
        magnitude = float(payload.get('magnitude', 0.0))
        depth = float(payload.get('depth', 10.0))
        dist = float(payload.get('distance_to_coast_km', 500.0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid numeric inputs"}), 400

    res = predict_tsunami(magnitude=magnitude, depth=depth, distance_to_coast_km=dist)
    return jsonify(res)

def _is_valid_latlon(lat, lon):
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

# ---------------------------
# Main create_report endpoint — modified to include Gemini + fusion + ETA
# ---------------------------

@app.route('/reports', methods=['POST'])
def create_report():
    """
    CitizenSOS report endpoint with Gemini fusion and ETA using ORS.
    POST JSON:
    { "message": "...", "lat": <float|null>, "lon": <float|null>, "user_id": optional }
    """
    payload = request.get_json() or {}
    message = (payload.get('message') or "").strip()
    raw_lat = payload.get('lat')
    raw_lon = payload.get('lon')
    user_id = payload.get('user_id')

    if not message:
        return jsonify({"status":"error","detail":"missing message"}), 400

    # Validate provided GPS if present
    if raw_lat is not None and raw_lon is not None:
        if not _is_valid_latlon(raw_lat, raw_lon):
            return jsonify({"status":"error","detail":"invalid lat/lon ranges"}), 400

    # 1) compute emergency score (heuristic)
    urgency = heuristic_emergency_score(message)
    is_emergency = 1 if urgency >= 0.5 else 0

    # 2) resolve location (same robust strategy)
    resolved_lat = None
    resolved_lon = None
    location_text = None

    if raw_lat is not None and raw_lon is not None:
        try:
            resolved_lat = float(raw_lat)
            resolved_lon = float(raw_lon)
            location_text = "user-provided GPS"
        except Exception:
            resolved_lat = resolved_lon = None

    if resolved_lat is None or resolved_lon is None:
        message_l = message.lower()
        bias_country = None
        if any(s in message_l for s in [" india", "india,", "kerala", "maharashtra", "tamil nadu", "delhi", "karnataka", "andhra", "telangana", "goa", "punjab", "uttar pradesh", "uttarakhand"]):
            bias_country = "in"

        candidates = extract_place_candidates(message)
        try:
            app.logger.debug(f"Place candidates: {candidates}, country_bias={bias_country}")
        except Exception:
            print(f"[geocode] Place candidates: {candidates}, country_bias={bias_country}")

        for c in candidates:
            tokens = re.findall(r'\w+', c)
            if len(tokens) == 1 and tokens[0].lower() in QUESTION_WORDS:
                try:
                    app.logger.debug(f"Skipping single-word non-place candidate: {c}")
                except Exception:
                    print(f"[geocode] Skipping single-word non-place candidate: {c}")
                continue

            geo = geocode_place_nominatim(c, countrycodes=bias_country)
            if geo:
                resolved_lat, resolved_lon, display = geo
                location_text = display
                break

            if bias_country is None and len(c.split()) <= 3:
                geo = geocode_place_nominatim(f"{c}, India", countrycodes="in")
                if geo:
                    resolved_lat, resolved_lon, display = geo
                    location_text = display
                    break

        if resolved_lat is None or resolved_lon is None:
            short_msg = message if len(message) <= 120 else message[:120]
            try:
                app.logger.debug(f"Attempting geocode fallback on message: {short_msg} (bias={bias_country})")
            except Exception:
                print(f"[geocode] Attempting geocode fallback on message: {short_msg} (bias={bias_country})")
            geo = geocode_place_nominatim(short_msg, countrycodes=bias_country)
            if geo:
                resolved_lat, resolved_lon, display = geo
                location_text = display

        if resolved_lat is None or resolved_lon is None:
            title_candidates = re.findall(r'\b[A-Z][a-z]{2,}\b', message)
            seen = set()
            filtered = []
            for t in title_candidates:
                tl = t.lower().strip()
                if tl in seen:
                    continue
                seen.add(tl)
                if tl in GENERIC_TOKENS or tl in PLACE_STOPWORDS:
                    continue
                if len(tl) < 3:
                    continue
                filtered.append(t)
            for tc in filtered:
                try:
                    app.logger.debug(f"Trying title-token geocode: {tc} (country=in)")
                except Exception:
                    print(f"[geocode] Trying title-token geocode: {tc} (country=in)")
                geo = geocode_place_nominatim(tc, countrycodes="in")
                if geo:
                    resolved_lat, resolved_lon, display = geo
                    location_text = display
                    break

        if resolved_lat is None or resolved_lon is None:
            words = re.findall(r'\w+', message)
            n = len(words)
            for k in range(1, min(4, n+1)):
                candidate = " ".join(words[-k:]).strip()
                if not candidate:
                    continue
                try:
                    app.logger.debug(f"Trying last-words geocode: '{candidate}, India'")
                except Exception:
                    print(f"[geocode] Trying last-words geocode: '{candidate}, India'")
                geo = geocode_place_nominatim(f"{candidate}, India", countrycodes="in")
                if geo:
                    resolved_lat, resolved_lon, display = geo
                    location_text = display
                    break

        if resolved_lat is None or resolved_lon is None:
            try:
                app.logger.debug("All geocoding fallbacks failed; resolved_lat still None")
            except Exception:
                print("[geocode] All geocoding fallbacks failed; resolved_lat still None")

    # 3) ML predictions + keyword fallback saved to ml_info
    ml_info = {c: 0 for c in category_names}
    try:
        cleaned = " ".join(tokenize(message))
    except Exception:
        cleaned = clean_text(message)

    # local model inference (robust handling)
    try:
        if model is not None and hasattr(model, "predict_proba"):
            raw = model.predict_proba([cleaned])
            if isinstance(raw, list):
                probs = list(raw[0]) if len(raw) and isinstance(raw[0], (list, tuple)) else [float(raw[0])]
            else:
                try:
                    probs = list(raw[0])
                except Exception:
                    probs = list(raw)
            threshold = 0.5
            for i, cname in enumerate(category_names):
                try:
                    p = float(probs[i]) if i < len(probs) else 0.0
                except Exception:
                    p = 0.0
                ml_info[cname] = 1 if p >= threshold else 0
        elif model is not None:
            preds = model.predict([cleaned])[0]
            for i, cname in enumerate(category_names):
                try:
                    ml_info[cname] = int(preds[i])
                except Exception:
                    ml_info[cname] = int(preds[i]) if isinstance(preds, (list, tuple)) and i < len(preds) else 0
        else:
            pass
    except Exception:
        ml_info = {c: 0 for c in category_names}

    # keyword fallback ensures explicit textual mentions are captured
    kw_text = message.lower()
    for cat, keys in keyword_map.items():
        if cat not in ml_info:
            ml_info.setdefault(cat, 0)
        for k in keys:
            if k in kw_text:
                ml_info[cat] = 1
                break

    # NEW: at least one category
    if not any(ml_info.values()):
        if "related" in ml_info:
            ml_info["related"] = 1
        else:
            first_cat = category_names[0] if category_names else None
            if first_cat:
                ml_info[first_cat] = 1

    # if certain categories present, promote emergency flag
    if any(ml_info.get(c) for c in ("medical_help", "request", "aid_related", "fire")):
        is_emergency = 1

    # ---------------------------
    # NEW: Gemini call and fusion logic
    # ---------------------------
    # Determine local top prediction and a local probability estimate (best-effort)
    local_top_label = "other"
    local_top_prob = 0.0
    try:
        if model is not None:
            # First try predict_proba robustly
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned])
                probs = None
                # Try array-like (n_samples, n_classes)
                try:
                    probs = list(proba[0])
                except Exception:
                    # maybe list of arrays (multioutput) - find candidate
                    if isinstance(proba, (list, tuple)) and len(proba) > 0:
                        for elt in proba:
                            try:
                                cand = list(elt[0]) if hasattr(elt, "__len__") else None
                                if cand and len(cand) > 1:
                                    probs = cand
                                    break
                            except Exception:
                                continue
                classes = getattr(model, "classes_", None)
                if probs is not None and classes is not None and len(classes) == len(probs):
                    idx = int(np.argmax(probs))
                    local_top_label = str(classes[idx])
                    local_top_prob = float(probs[idx])
                elif probs is not None:
                    idx = int(np.argmax(probs))
                    local_top_label = f"pred_{idx}"
                    local_top_prob = float(probs[idx])
                else:
                    # fallback to predict()
                    try:
                        pred = model.predict([cleaned])
                        local_top_label = str(pred[0]) if isinstance(pred, (list, tuple)) else str(pred)
                        local_top_prob = 0.75
                    except Exception:
                        local_top_label = "other"
                        local_top_prob = 0.2
            else:
                # no predict_proba
                pred = model.predict([cleaned])
                local_top_label = str(pred[0]) if isinstance(pred, (list, tuple)) else str(pred)
                local_top_prob = 0.75
        else:
            # no model: use keyword heuristics
            if "collapse" in kw_text or "collapsed" in kw_text:
                local_top_label = "buildings"
                local_top_prob = 0.7
            else:
                local_top_label = "other"
                local_top_prob = 0.2
    except Exception as e:
        app.logger.warning(f"Error deriving local_top (robust): {e}")
        local_top_label = "other"
        local_top_prob = 0.2

    # call Gemini (may return None)
    try:
        gemini_result = gemini_analyze_text(message)
        if gemini_result is None:
            app.logger.info("Gemini returned None (unavailable or parsing failed).")
    except Exception as e:
        app.logger.warning(f"Gemini call failed: {e}")
        gemini_result = None

    # compute a scalar keyword_score (simple)
    keyword_score = 0.0
    for k, w in keyword_map.items():
        for token in w:
            if token in kw_text:
                keyword_score += 0.02
    keyword_score = min(1.0, keyword_score)

    # fuse into final label + confidence
    fused = fuse_with_gemini(local_top_label, local_top_prob, keyword_score, gemini_result)
    fused_label = fused.get("label", local_top_label)
    fused_confidence = fused.get("confidence", 0.0)

    # ------------------------------
    # Build ml_record including people count (from Gemini)
    # ------------------------------
    # gemini_result may be None or a dict; safe-extract people
    people_count = None
    try:
        if isinstance(gemini_result, dict):
            people_val = gemini_result.get("people", None)
            if isinstance(people_val, (int, float)):
                people_count = int(people_val)
            elif isinstance(people_val, str) and people_val.strip().isdigit():
                people_count = int(people_val.strip())
            else:
                people_count = gemini_result.get("people")  # could be None
    except Exception:
        people_count = None

    ml_record = {
        "ml_info": ml_info,
        "local_top": {"label": local_top_label, "prob": local_top_prob},
        "gemini": gemini_result,
        "fused": fused,
        "people": people_count
    }

    # thresholds (will be adjusted by people_count below)
    AUTO_DISPATCH_THRESHOLD = 0.90
    HUMAN_TRIAGE_THRESHOLD = 0.60

    # escalate if many people affected
    if ml_record.get("people") is not None:
        try:
            ppl = int(ml_record.get("people"))
        except Exception:
            ppl = 0
        if ppl >= 5:
            AUTO_DISPATCH_THRESHOLD = 0.80
            HUMAN_TRIAGE_THRESHOLD = 0.45
        elif ppl >= 2:
            AUTO_DISPATCH_THRESHOLD = 0.875
            HUMAN_TRIAGE_THRESHOLD = 0.55

    # Auto-dispatch override: if 3+ people and confidence moderate, escalate to auto_dispatch
    if ml_record.get("people") is not None:
        try:
            ppl_chk = int(ml_record.get("people"))
        except Exception:
            ppl_chk = 0
        if ppl_chk >= 3 and fused_confidence >= 0.65:
            # immediate override to auto_dispatch
            action = "auto_dispatch"
        else:
            # Safer dispatch decision:
            if fused_confidence >= AUTO_DISPATCH_THRESHOLD:
                if fused_label == "other":
                    # only auto if gemini explicitly says it's an emergency
                    if gemini_result and isinstance(gemini_result, dict) and gemini_result.get("is_emergency"):
                        action = "auto_dispatch"
                    else:
                        app.logger.info("Auto-dispatch blocked because fused_label='other' and Gemini not confirming.")
                        action = "human_triage"
                else:
                    action = "auto_dispatch"
            elif fused_confidence >= HUMAN_TRIAGE_THRESHOLD:
                action = "human_triage"
            else:
                action = "clarify"
    else:
        # no people info: use normal thresholds
        if fused_confidence >= AUTO_DISPATCH_THRESHOLD:
            if fused_label == "other":
                if gemini_result and isinstance(gemini_result, dict) and gemini_result.get("is_emergency"):
                    action = "auto_dispatch"
                else:
                    app.logger.info("Auto-dispatch blocked because fused_label='other' and Gemini not confirming.")
                    action = "human_triage"
            else:
                action = "auto_dispatch"
        elif fused_confidence >= HUMAN_TRIAGE_THRESHOLD:
            action = "human_triage"
        else:
            action = "clarify"

    # Persist report to DB (including ml_info + gemini + fused)
    assigned_helpers = []
    created_at = datetime.utcnow().isoformat() + "Z"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO reports (message, raw_lat, raw_lon, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, ml_info, assigned_helpers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message, raw_lat, raw_lon, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, json.dumps(ml_record), json.dumps(assigned_helpers)))
        report_rowid = cur.lastrowid
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"status":"error","detail":"db insert failed","error":str(e)}), 500

    # If emergency -> find helpers and notify (attach ETA info and numeric ORS fields)
    if is_emergency:
        helpers = find_nearby_helpers(resolved_lat, resolved_lon, max_results=5, max_radius_km=15)
        for h in helpers:
            # compute ETA using ORS (cached) with fallback to haversine rough estimate
            eta = None
            eta_info = None
            try:
                if resolved_lat is not None and resolved_lon is not None and h.get('lat') is not None and h.get('lon') is not None:
                    eta_info = get_eta_openrouteservice_cached(resolved_lat, resolved_lon, h['lat'], h['lon'])
                if eta_info and eta_info.get("duration_text"):
                    eta = format_duration_text(eta_info.get("duration_text"), duration_s=eta_info.get("duration_s"))
                else:
                    dist = haversine(resolved_lat, resolved_lon, h['lat'], h['lon'])
                    eta = get_eta_approx_km_to_minutes(dist)
            except Exception:
                eta = "unknown"

            # notify (email) - pass eta_text so notify_helper_via_email can use it
            res_send = notify_helper_via_email(h, {
                "message": message,
                "short_msg": message[:140],
                "urgency": urgency,
                "resolved_lat": resolved_lat,
                "resolved_lon": resolved_lon,
                "location_text": location_text
            }, eta_text=eta)

            # attach numeric ORS values if present
            eta_seconds = None
            eta_distance_km = None
            try:
                if eta_info:
                    eta_seconds = eta_info.get("duration_s")
                    eta_distance_km = eta_info.get("distance_km")
                else:
                    tmp = get_eta_openrouteservice_cached(resolved_lat, resolved_lon, h['lat'], h['lon'])
                    if tmp:
                        eta_seconds = tmp.get("duration_s")
                        eta_distance_km = tmp.get("distance_km")
            except Exception:
                eta_seconds = None
                eta_distance_km = None

            assigned_helpers.append({
                'helper_id': h['id'],
                'name': h['name'],
                'email': h['email'],
                'dist_km': h['dist_km'],
                'eta': eta,
                'eta_seconds': eta_seconds,
                'eta_distance_km': round(eta_distance_km, 3) if eta_distance_km is not None else None,
                'send_result': res_send
            })
        # write assigned helpers to DB
        try:
            cur.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(assigned_helpers), report_rowid))
            conn.commit()
        except Exception:
            pass

    conn.close()

    # build response payload (including Gemini/ML info and fused decision)
    response_payload = {
        "status": "ok",
        "report_id": report_rowid,
        "is_emergency": is_emergency,
        "urgency": urgency,
        "resolved_lat": resolved_lat,
        "resolved_lon": resolved_lon,
        "location_text": location_text,
        "assigned_helpers": assigned_helpers,
        "ml_info": ml_record,
        "fused_label": fused_label,
        "fused_confidence": fused_confidence,
        "action": action,
        "people": ml_record.get("people")
    }
    # include Gemini summary if present
    if gemini_result and isinstance(gemini_result, dict):
        response_payload["summary"] = gemini_result.get("summary")
        response_payload["urgency_label"] = gemini_result.get("urgency")
        response_payload["gemini"] = gemini_result

    return jsonify(response_payload)

# alerts endpoints unchanged
@app.route('/alerts/send', methods=['POST'])
def api_send_alert():
    payload = request.get_json() or {}
    to = payload.get('to')
    subject = payload.get('subject', 'Disaster Alert')
    body = payload.get('body', 'This is an alert from DisasterGuard')
    if not to:
        return jsonify({'status': 'error', 'detail': 'missing recipient'}), 400

    # send the email (existing function)
    result = send_email(to_email=to, subject=subject, body=body)

    # build audit record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "to": to,
        "subject": subject,
        "body_snippet": (body[:200] + '...') if len(body) > 200 else body,
        "send_result": result
    }

    # append as one JSON-per-line to the log file
    try:
        with open(ALERT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        result = {"status": "error", "detail": f"email send result: {result}; log error: {str(e)}"}

    return jsonify(result)


@app.route('/alerts/logs', methods=['GET'])
def api_get_alert_logs():
    try:
        n = int(request.args.get('n', 20))
    except Exception:
        n = 20

    if not os.path.exists(ALERT_LOG_PATH):
        return jsonify([])

    records = []
    try:
        with open(ALERT_LOG_PATH, 'r', encoding='utf-8') as f:
            last_lines = deque(f, maxlen=n)
            for line in last_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    records.append({"raw": line})
    except Exception:
        return jsonify([])

    records.reverse()
    return jsonify(records)

# other helper endpoints unchanged (recent reports, resolve, assign, export)
@app.route('/reports/recent', methods=['GET'])
def api_get_recent_reports():
    try:
        n = int(request.args.get('n', 25))
    except Exception:
        n = 25
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, message, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, assigned_helpers, ml_info FROM reports ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "message": r[1],
            "resolved_lat": r[2],
            "resolved_lon": r[3],
            "location_text": r[4],
            "urgency": r[5],
            "is_emergency": r[6],
            "created_at": r[7],
            "assigned_helpers": json.loads(r[8]) if r[8] else [],
            "ml_info": json.loads(r[9]) if r[9] else {}
        })
    return jsonify(out)

@app.route('/report/<int:report_id>/resolve', methods=['POST'])
def api_resolve_report(report_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        resolved_at = datetime.utcnow().isoformat() + "Z"
        try:
            cur.execute("PRAGMA table_info(reports)")
            cols = [r[1] for r in cur.fetchall()]
            if 'resolved' not in cols:
                cur.execute("ALTER TABLE reports ADD COLUMN resolved INTEGER DEFAULT 0")
            if 'resolved_at' not in cols:
                cur.execute("ALTER TABLE reports ADD COLUMN resolved_at TEXT")
            conn.commit()
        except Exception:
            pass
        cur.execute("UPDATE reports SET resolved = 1, resolved_at = ? WHERE id = ?", (resolved_at, report_id))
        conn.commit()
        conn.close()
        return jsonify({"status":"ok","report_id":report_id,"resolved_at":resolved_at})
    except Exception as e:
        return jsonify({"status":"error","detail":str(e)}), 500

@app.route('/report/<int:report_id>/assign', methods=['POST'])
def api_assign_helper(report_id):
    payload = request.get_json() or {}
    helper_id = payload.get('helper_id')
    if helper_id is None:
        return jsonify({"status":"error","detail":"missing helper_id"}), 400
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, lat, lon FROM helpers WHERE id = ?", (helper_id,))
        h = cur.fetchone()
        if not h:
            conn.close()
            return jsonify({"status":"error","detail":"helper not found"}), 404
        hid, name, email, hlat, hlon = h
        cur.execute("SELECT assigned_helpers FROM reports WHERE id = ?", (report_id,))
        row = cur.fetchone()
        existing = []
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except Exception:
                existing = []
        cur.execute("SELECT resolved_lat, resolved_lon FROM reports WHERE id = ?", (report_id,))
        rr = cur.fetchone()
        dist_km = None
        if rr and rr[0] is not None and rr[1] is not None:
            try:
                dist_km = haversine(float(rr[0]), float(rr[1]), float(hlat), float(hlon))
                dist_km = round(dist_km, 2)
            except Exception:
                dist_km = None

        # compute ETA for manual assign using ORS cached if possible
        eta = None
        eta_seconds = None
        eta_distance_km = None
        try:
            if rr and rr[0] is not None and rr[1] is not None and hlat is not None and hlon is not None:
                eta_info = get_eta_openrouteservice_cached(rr[0], rr[1], hlat, hlon)
                if eta_info and eta_info.get("duration_s") is not None:
                    eta = format_duration_text(duration_s=eta_info.get("duration_s"))
                    eta_seconds = eta_info.get("duration_s")
                    eta_distance_km = eta_info.get("distance_km")
                elif eta_info and eta_info.get("duration_text"):
                    eta = format_duration_text(eta_info.get("duration_text"))
                    eta_seconds = eta_info.get("duration_s")
                    eta_distance_km = eta_info.get("distance_km")
                else:
                    if dist_km is not None:
                        eta = get_eta_approx_km_to_minutes(dist_km)
        except Exception:
            eta = None

        new = {
            "helper_id": hid,
            "name": name,
            "email": email,
            "dist_km": dist_km,
            "eta": eta,
            "eta_seconds": eta_seconds,
            "eta_distance_km": round(eta_distance_km, 3) if eta_distance_km is not None else None
        }
        existing.append(new)
        cur.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(existing), report_id))
        conn.commit()
        conn.close()
        return jsonify({"status":"ok","report_id":report_id,"assigned": new})
    except Exception as e:
        return jsonify({"status":"error","detail":str(e)}), 500

# Add this to run.py (near other report endpoints)
@app.route('/report/<int:report_id>/recompute_eta', methods=['POST'])
def api_recompute_eta(report_id):
    """
    Recompute ORS ETA for assigned helpers of the given report.
    Returns updated assigned_helpers list (with eta, eta_seconds, eta_distance_km).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT resolved_lat, resolved_lon, assigned_helpers FROM reports WHERE id = ?", (report_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"status":"error","detail":"report not found"}), 404
        resolved_lat, resolved_lon, assigned_json = row
        if resolved_lat is None or resolved_lon is None:
            conn.close()
            return jsonify({"status":"error","detail":"report has no resolved coordinates"}), 400

        assigned = []
        if assigned_json:
            try:
                assigned = json.loads(assigned_json)
            except Exception:
                assigned = []

        updated = []
        for h in assigned:
            hlat = h.get('lat') or h.get('lat')  # depends on your stored helper structure
            hlon = h.get('lon') or h.get('lon')
            # If not present in stored helper dict, fetch helper row
            if hlat is None or hlon is None:
                try:
                    cur.execute("SELECT lat, lon FROM helpers WHERE id = ?", (h.get('helper_id'),))
                    hr = cur.fetchone()
                    if hr:
                        hlat, hlon = float(hr[0]), float(hr[1])
                except Exception:
                    hlat = hlon = None

            eta_text = h.get('eta')
            eta_seconds = None
            eta_distance_km = None
            if hlat is not None and hlon is not None:
                # Force fresh ORS call (bypass cache)
                fresh = get_eta_openrouteservice(resolved_lat, resolved_lon, hlat, hlon)
                if fresh:
                    eta_seconds = fresh.get("duration_s")
                    eta_distance_km = fresh.get("distance_km")
                    try:
                        eta_text = format_duration_text(duration_s=eta_seconds)
                    except Exception:
                        eta_text = fresh.get("duration_text") or eta_text

            h['eta'] = eta_text
            h['eta_seconds'] = eta_seconds
            h['eta_distance_km'] = round(eta_distance_km,3) if eta_distance_km is not None else None
            updated.append(h)

        # persist updated assigned_helpers back to DB
        try:
            cur.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(updated), report_id))
            conn.commit()
        except Exception:
            app.logger.warning("Could not persist recomputed assigned_helpers for report %s", report_id)
        conn.close()
        return jsonify({"status":"ok","assigned_helpers": updated})
    except Exception as e:
        app.logger.exception("recompute_eta error")
        return jsonify({"status":"error","detail":str(e)}), 500

@app.route('/reports/export', methods=['GET'])
def api_export_reports_csv():
    try:
        n = int(request.args.get('n', 200))
    except Exception:
        n = 200
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, message, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, assigned_helpers, ml_info FROM reports ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
        conn.close()
    except Exception:
        rows = []
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id','message','location_text','resolved_lat','resolved_lon','urgency','is_emergency','created_at','assigned_helpers','ml_info'])
    for r in rows:
        writer.writerow([r[0], r[1] or '', r[4] or '', r[2] or '', r[3] or '', r[5] or '', r[6] or 0, r[7] or '', r[8] or '', r[9] or ''])
    csv_str = buf.getvalue()
    from flask import Response
    return Response(csv_str, mimetype='text/csv', headers={"Content-disposition":"attachment; filename=reports_export.csv"})

@app.route('/dashboard')
def dashboard():
    import sqlite3, os, json
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "DisasterResponse.db")
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, message, resolved_lat, resolved_lon, location_text, urgency, is_emergency, assigned_helpers, ml_info, created_at
        FROM reports
        ORDER BY id DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    conn.close()

    reports = []
    for r in rows:
        helpers = []
        try:
            helpers = json.loads(r[7]) if r[7] else []
        except:
            helpers = []
        ml_info = {}
        try:
            ml_info = json.loads(r[8]) if r[8] else {}
        except:
            ml_info = {}

        preds = []
        try:
            if ml_info:
                for cname in category_names:
                    if ml_info.get(cname):
                        preds.append(cname)
            for cname, val in ml_info.items():
                if val and cname not in preds:
                    preds.append(cname)
        except Exception:
            preds = []

        if not preds:
            preds_display = []
        else:
            preds_display = preds

        reports.append({
            "id": r[0],
            "message": r[1],
            "lat": r[2],
            "lon": r[3],
            "location": r[4],
            "urgency": r[5],
            "is_emergency": r[6],
            "helpers": helpers,
            "ml_info": ml_info,
            "predicted_categories": preds_display,
            "created_at": r[9] if len(r) > 9 else "N/A",
            "assigned_helpers": helpers  # include numeric fields for dashboard
        })

    return render_template("dashboard.html", reports=reports)

@app.route('/helpers', methods=['POST'])
def api_add_helper():
    p = request.get_json() or {}
    name = p.get('name'); email = p.get('email'); phone = p.get('phone')
    lat = p.get('lat'); lon = p.get('lon'); radius = p.get('radius_km', 10)
    if not name or lat is None or lon is None:
        return jsonify({"status":"error","detail":"name, lat and lon required"}), 400
    if not _is_valid_latlon(lat, lon):
        return jsonify({"status":"error","detail":"invalid lat/lon ranges"}), 400
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO helpers (name,email,phone,lat,lon,radius_km,active) VALUES (?,?,?,?,?,?,1)",
                    (name, email, phone, float(lat), float(lon), float(radius)))
        conn.commit(); hid = cur.lastrowid
    except Exception as e:
        conn.close()
        return jsonify({"status":"error","detail":"db insert failed","error":str(e)}), 500
    conn.close()
    return jsonify({"status":"ok","helper_id": hid})

@app.route('/report')
def report_page():
    prefill = request.args.get("prefill", "")
    return render_template('report.html', prefill=prefill)

# ---------------------------
# Live Maharashtra monitoring API
# ---------------------------
from datetime import timezone
from functools import lru_cache
import csv
import io

LIVE_CITIES = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Nashik", "lat": 19.9975, "lon": 73.7898},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Chhatrapati Sambhajinagar", "lat": 19.8762, "lon": 75.3433},
]
_LIVE_CACHE = {"value": None, "expires": 0}
_LIVE_CACHE_SECONDS = 240  # provider fetch cache; browser refreshes every five minutes


def _fetch_open_meteo(city):
    """Fetch current conditions and short-range hourly forecast; values are provider data."""
    params = {
        "latitude": city["lat"], "longitude": city["lon"],
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "hourly": "precipitation,temperature_2m,wind_speed_10m",
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=12)
    r.raise_for_status()
    payload = r.json()
    current = payload.get("current", {})
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    precip = hourly.get("precipitation", [])
    temp = hourly.get("temperature_2m", [])
    wind = hourly.get("wind_speed_10m", [])
    # Keep the next 24 hourly forecast points from the provider's current timestamp.
    start = 0
    current_time = current.get("time")
    if current_time and current_time in times:
        start = times.index(current_time)
    series = []
    for i in range(start, min(start + 24, len(times))):
        series.append({
            "time": times[i],
            "precipitation_mm": precip[i] if i < len(precip) else None,
            "temperature_c": temp[i] if i < len(temp) else None,
            "wind_kmh": wind[i] if i < len(wind) else None,
        })
    return {
        "city": city["name"], "latitude": city["lat"], "longitude": city["lon"],
        "current": current, "hourly_forecast": series,
        "source": "Open-Meteo forecast API",
        "source_url": "https://open-meteo.com/",
        "timezone": payload.get("timezone", "Asia/Kolkata")
    }


def _fetch_firms():
    """Fetch latest NASA FIRMS VIIRS hotspots for Maharashtra if a free MAP_KEY is configured."""
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        return {"available": False, "message": "FIRMS_MAP_KEY is not configured. Add a free NASA FIRMS MAP_KEY to .env to enable fire detections.", "features": [], "source": "NASA FIRMS", "source_url": "https://firms.modaps.eosdis.nasa.gov/"}
    # Approximate Maharashtra bounding box: west,south,east,north
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_NOAA20_NRT/72.5,15.5,81.0,22.1/2"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
        if text.lstrip().lower().startswith(("invalid", "error", "please")):
            raise ValueError("NASA FIRMS did not return CSV data; check the MAP_KEY and quota.")
        rows = csv.DictReader(io.StringIO(text))
        features = []
        for row in rows:
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                if not (15.5 <= lat <= 22.1 and 72.5 <= lon <= 81.0):
                    continue
                features.append({
                    "latitude": lat, "longitude": lon,
                    "acq_date": row.get("acq_date"), "acq_time": row.get("acq_time"),
                    "confidence": row.get("confidence"), "frp": _safe_float(row.get("frp")),
                    "satellite": row.get("satellite"), "instrument": row.get("instrument"),
                    "daynight": row.get("daynight"), "source": "NASA FIRMS VIIRS NOAA-20 NRT"
                })
            except (TypeError, ValueError, KeyError):
                continue
        return {"available": True, "features": features[:3000], "count": len(features), "source": "NASA FIRMS VIIRS NOAA-20 NRT", "source_url": "https://firms.modaps.eosdis.nasa.gov/api/area", "fetched_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        app.logger.warning("NASA FIRMS fetch failed: %s", exc)
        return {"available": False, "message": "NASA FIRMS is temporarily unavailable or rejected the request. Check the MAP_KEY, quota and server logs.", "features": [], "source": "NASA FIRMS", "source_url": "https://firms.modaps.eosdis.nasa.gov/api/area"}


def _safe_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


@app.route('/api/live-data', methods=['GET'])
def api_live_data():
    """Return real provider data and explicit per-source availability/errors."""
    now = time.time()
    if _LIVE_CACHE["value"] is not None and now < _LIVE_CACHE["expires"]:
        cached = dict(_LIVE_CACHE["value"])
        cached["cache"] = {"hit": True, "refresh_after_seconds": max(0, int(_LIVE_CACHE["expires"] - now))}
        return jsonify(cached)

    fetched_at = datetime.now(timezone.utc).isoformat()
    weather, weather_errors = [], []
    for city in LIVE_CITIES:
        try:
            weather.append(_fetch_open_meteo(city))
        except Exception as exc:
            app.logger.warning("Open-Meteo fetch failed for %s: %s", city["name"], exc)
            weather_errors.append(city["name"])
    fires = _fetch_firms()
    response = {
        "region": "Maharashtra, India",
        "fetched_at": fetched_at,
        "refresh_interval_seconds": 300,
        "weather": {"available": bool(weather), "cities": weather,
                    "errors": weather_errors,
                    "source": "Open-Meteo current conditions + hourly forecast",
                    "source_url": "https://open-meteo.com/en/docs"},
        "fires": fires,
        "rainfall": {
            "available": True,
            "mode": "map_overlay",
            "source": "NASA GPM IMERG precipitation-rate imagery via NASA GIBS",
            "source_url": "https://gpm.nasa.gov/data/imerg",
            "tile_url": "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
            "layer": "IMERG_Precipitation_Rate",
            "note": "Satellite precipitation imagery is shown as a map overlay. It is not a point-level rainfall time series in this first version; hourly precipitation charts below are Open-Meteo forecast values and are labelled as forecasts."
        },
        "sources": [
            {"name": "Open-Meteo", "status": "available" if weather else "error", "url": "https://open-meteo.com/"},
            {"name": "NASA FIRMS", "status": "available" if fires.get("available") else "needs_key_or_error", "url": "https://firms.modaps.eosdis.nasa.gov/"},
            {"name": "NASA GPM IMERG via GIBS", "status": "map_overlay", "url": "https://gpm.nasa.gov/data/imerg"}
        ]
    }
    _LIVE_CACHE["value"] = response
    _LIVE_CACHE["expires"] = now + _LIVE_CACHE_SECONDS
    response["cache"] = {"hit": False, "refresh_after_seconds": _LIVE_CACHE_SECONDS}
    return jsonify(response)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

import endpoints

# --- Added endpoints for frontend preview and live ETA polling ---
import time, json, sqlite3, math

def _haversine(lat1, lon1, lat2, lon2):
    if None in (lat1,lon1,lat2,lon2):
        return float('inf')
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

_ETA_CACHE = {}
_ETA_CACHE_TTL = 60
def _cache_key_for_eta(o_lat, o_lon, d_lat, d_lon, mode='driving'):
    return f"{float(o_lat):.6f},{float(o_lon):.6f}->{float(d_lat):.6f},{float(d_lon):.6f}|{mode}"

def _get_eta_cached(o_lat, o_lon, d_lat, d_lon):
    key = _cache_key_for_eta(o_lat,o_lon,d_lat,d_lon)
    now = time.time()
    if key in _ETA_CACHE:
        ts, val = _ETA_CACHE[key]
        if now - ts < _ETA_CACHE_TTL:
            return val
    # Try to use existing get_eta_openrouteservice if available
    try:
        val = get_eta_openrouteservice(o_lat, o_lon, d_lat, d_lon)
    except Exception:
        val = None
    _ETA_CACHE[key] = (now, val)
    return val

def _format_duration_text(duration_s=None, duration_text=None):
    try:
        if duration_s is not None:
            if duration_s < 60:
                return "<1 min"
            mins = int(round(duration_s/60.0))
            return f"{mins} min" if mins == 1 else f"{mins} mins"
        if duration_text:
            return duration_text
    except Exception:
        pass
    return "unknown"

@app.route('/find_nearby', methods=['POST'])
def api_find_nearby():
    payload = request.get_json() or {}
    try:
        lat = float(payload.get('lat'))
        lon = float(payload.get('lon'))
    except Exception:
        return jsonify({"status":"error","detail":"lat and lon required numeric"}), 400
    max_radius_km = float(payload.get('max_radius_km', 15))
    max_results = int(payload.get('max_results', 6))
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT id, name, email, phone, lat, lon, radius_km FROM helpers WHERE active=1")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        rows = []
    helpers = []
    for r in rows:
        try:
            hid, name, email, phone, hlat, hlon, hradius = r
        except Exception:
            continue
        if hlat is None or hlon is None:
            continue
        dist = _haversine(lat, lon, hlat, hlon)
        allowed = hradius if (hradius and hradius>0) else max_radius_km
        if dist <= allowed:
            eta_info = _get_eta_cached(lat, lon, hlat, hlon)
            eta_text = None; eta_seconds = None
            if eta_info:
                eta_seconds = eta_info.get("duration_s"); eta_text = _format_duration_text(duration_s=eta_seconds, duration_text=eta_info.get("duration_text"))
            else:
                mins = max(1, int(round((dist / 35.0) * 60)))
                eta_text = f"{mins} mins"
            helpers.append({
                "helper_id": hid, "name": name, "email": email, "phone": phone,
                "lat": hlat, "lon": hlon, "dist_km": round(dist,2),
                "eta_text": eta_text, "eta_seconds": eta_seconds
            })
    helpers = sorted(helpers, key=lambda x: x.get('dist_km') or 999)[:max_results]
    return jsonify({"status":"ok","helpers":helpers})

@app.route('/eta_status/<int:report_id>', methods=['GET'])
def api_eta_status(report_id):
    bypass_cache = request.args.get('bypass_cache','false').lower() in ('1','true','yes')
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        cur.execute("SELECT assigned_helpers, resolved_lat, resolved_lon FROM reports WHERE id = ?", (report_id,))
        row = cur.fetchone()
    except Exception:
        row = None
    conn.close()
    if not row:
        return jsonify({"status":"error","detail":"report not found"}), 404
    assigned_json, resolved_lat, resolved_lon = row
    assigned = []
    if assigned_json:
        try: assigned = json.loads(assigned_json)
        except Exception: assigned = []
    updated = []
    for h in assigned:
        hlat = h.get('lat'); hlon = h.get('lon')
        if (hlat is None or hlon is None) and h.get('helper_id'):
            try:
                conn2 = sqlite3.connect(DB_PATH); cur2 = conn2.cursor()
                cur2.execute("SELECT lat, lon FROM helpers WHERE id = ?", (h.get('helper_id'),))
                hr = cur2.fetchone(); conn2.close()
                if hr: hlat, hlon = float(hr[0]), float(hr[1])
            except Exception:
                hlat = hlon = None
        eta_text = h.get('eta'); eta_seconds = h.get('eta_seconds')
        if resolved_lat is not None and resolved_lon is not None and hlat is not None and hlon is not None:
            try:
                eta_info = _get_eta_cached(resolved_lat, resolved_lon, hlat, hlon)
                if eta_info:
                    eta_seconds = eta_info.get("duration_s")
                    eta_text = _format_duration_text(duration_s=eta_seconds, duration_text=eta_info.get("duration_text"))
                else:
                    dist = _haversine(resolved_lat, resolved_lon, hlat, hlon)
                    mins = max(1, int(round((dist / 35.0) * 60)))
                    eta_text = f"{mins} mins"
            except Exception:
                pass
        h['eta'] = eta_text; h['eta_seconds'] = eta_seconds
        updated.append(h)
    try:
        conn3 = sqlite3.connect(DB_PATH); cur3 = conn3.cursor()
        cur3.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(updated), report_id))
        conn3.commit(); conn3.close()
    except Exception:
        pass
    return jsonify({"status":"ok","report_id": report_id, "assigned_helpers": updated})
# --- end added endpoints ---

if __name__ == '__main__':
    main()
