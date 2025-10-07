# app/run.py
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

# helper modules (make sure you created these under src/)
from src.preprocess import clean_text
from src.predictors.earthquake import predict_quake
from src.predictors.flood import predict_flood
from src.predictors.tsunami import predict_tsunami
from src.alerts import send_email

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
    # You can expand this further with other categories similarly
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

# small stoplist for generic place words
PLACE_STOPWORDS = set([
    "building","house","home","here","nearby","near","site","place","area","location",
    "street","road","way","corner","block","shop","office","people","person","trapped","collapsed","collapse","injured","dead","help"
])

# generic tokens we should not try as place names (single-word)
GENERIC_TOKENS = set(["building","people","trapped","collapsed","collapse","injured","dead","help","here","nearby","home","where","what","who","when","why","how"])

# Words that alone are likely question tokens — avoid geocoding them
QUESTION_WORDS = set(["where","what","who","when","why","how"])

def extract_place_candidates(text):
    """Improved candidate extraction:
    - look for 'near|at|in' patterns,
    - split by commas and common separators,
    - include capitalized phrases and last 2-5 words as fallbacks.
    """
    text = (text or "").strip()
    if not text:
        return []

    # patterns like "near X", "at X", "in X"
    patterns = [r'near\s+([A-Za-z0-9\s\-\_\,]+)',
                r'at\s+([A-Za-z0-9\s\-\_\,]+)',
                r'in\s+([A-Za-z0-9\s\-\_\,]+)']
    candidates = []
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            m = m.strip().strip('.,;:')
            if len(m) > 2 and not re.match(r'^(me|home|here|nearby)$', m.strip(), re.I):
                candidates.append(m)

    # split by punctuation for explicit place mentions
    comma_parts = [part.strip() for part in re.split(r'[,\;\-\\/]', text) if len(part.strip())>3]
    candidates.extend(comma_parts)

    # include sequences of capitalized words (e.g., "MG Road", "Padukkad")
    caps = re.findall(r'([A-Z][a-z]{1,}\s(?:[A-Z][a-z]{1,}\s?){0,2})', text)
    candidates.extend([c.strip() for c in caps if len(c.strip())>2])

    # fallback: last 2–5 words
    words = re.findall(r'\w+', text)
    if len(words) >= 2:
        for n in range(2, min(6, len(words)+1)):
            candidates.append(" ".join(words[-n:]))

    # filter stopwords and dedupe
    seen = set(); out = []
    for c in candidates:
        key = c.lower().strip()
        tokens = [w for w in re.findall(r'\w+', key)]
        if len(tokens) == 0:
            continue
        # drop candidates that are only stopwords/generic tokens
        if all(t in PLACE_STOPWORDS for t in tokens):
            continue
        if key not in seen:
            seen.add(key); out.append(c)
    return out

def geocode_place_nominatim(place, countrycodes=None, timeout=8):
    """Geocode place using Nominatim. Returns (lat, lon, display_name) or None.
    Optional countrycodes: e.g., 'in' to bias to India.
    """
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

def notify_helper_via_email(helper, report):
    """Notify helper via email using src.alerts.send_email"""
    subj = f"URGENT: Nearby emergency — {report.get('short_msg','')[:80]}"
    maps_link = ""
    if report.get('resolved_lat') and report.get('resolved_lon'):
        maps_link = f"https://www.google.com/maps/search/?api=1&query={report['resolved_lat']},{report['resolved_lon']}"
    body = (
        "Hello {},\n\n"
        "A nearby emergency has been reported.\n\n"
        "Message: {}\n"
        "Urgency: {}\n"
        "Location: {}\n"
        "Map: {}\n\n"
        "If you can help, please reply to this email or follow local procedures.\n\n"
        "-- CitizenSOS"
    ).format(helper.get('name') or "Volunteer", report.get('message') or "", report.get('urgency'), report.get('location_text') or 'not available', maps_link)
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
# Flask routes
# ---------------------------

# index webpage displays cool visuals and receives user input text for model
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
                'xaxis': {'title': "Category"}
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
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
# New API endpoints (demo heuristics + alerting)
# ---------------------------

@app.route('/predict/earthquake', methods=['POST'])
def api_predict_quake():
    """
    POST JSON:
    { "lat": <float>, "lon": <float>, "depth": <float>, "date": "YYYY-MM-DD" }
    """
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
    """
    POST JSON:
    { "rainfall_mm": <float>, "dam_capacity_percent": <float> }
    """
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
    """
    POST JSON:
    { "magnitude": <float>, "depth": <float>, "distance_to_coast_km": <float> }
    """
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
    """Return True if lat/lon are valid numeric ranges."""
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


@app.route('/reports', methods=['POST'])
def create_report():
    """
    CitizenSOS report endpoint
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

    # 1) compute emergency score (heuristic for now)
    urgency = heuristic_emergency_score(message)
    is_emergency = 1 if urgency >= 0.5 else 0  # threshold 0.5; tune later

    # 2) resolve location
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
        # decide whether to bias to India
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
            # Skip single-word pure question tokens (e.g., "Where")
            tokens = re.findall(r'\w+', c)
            if len(tokens) == 1 and tokens[0].lower() in QUESTION_WORDS:
                try:
                    app.logger.debug(f"Skipping single-word non-place candidate: {c}")
                except Exception:
                    print(f"[geocode] Skipping single-word non-place candidate: {c}")
                continue

            # 1) try raw candidate with country bias (if available)
            geo = geocode_place_nominatim(c, countrycodes=bias_country)
            if geo:
                resolved_lat, resolved_lon, display = geo
                location_text = display
                break

            # 2) try candidate + ", India" if no bias and candidate is short
            if bias_country is None and len(c.split()) <= 3:
                geo = geocode_place_nominatim(f"{c}, India", countrycodes="in")
                if geo:
                    resolved_lat, resolved_lon, display = geo
                    location_text = display
                    break

        # fallback: geocode whole message (shortened) with India bias if available
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

        # --- Additional fallbacks: title tokens + last-word India guesses ---
        if resolved_lat is None or resolved_lon is None:
            # title-case tokens (Padukkad etc.) BUT filter out short/generic tokens
            title_candidates = re.findall(r'\b[A-Z][a-z]{2,}\b', message)
            seen = set()
            filtered = []
            for t in title_candidates:
                tl = t.lower().strip()
                if tl in seen:
                    continue
                seen.add(tl)
                # skip generic/common tokens
                if tl in GENERIC_TOKENS or tl in PLACE_STOPWORDS:
                    continue
                # skip very short tokens
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
            # try last 1-3 words appended with ', India'
            words = re.findall(r'\w+', message)
            n = len(words)
            for k in range(1, min(4, n+1)):  # 1..3
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
    # Ensure ml_info has all category_names with default 0
    ml_info = {c: 0 for c in category_names}

    try:
        # try to use same cleaning/tokenization as /go
        cleaned = " ".join(tokenize(message))
    except Exception:
        cleaned = clean_text(message)

    try:
        if model is not None and hasattr(model, "predict_proba"):
            raw = model.predict_proba([cleaned])
            # try to interpret raw into a list of probabilities
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
            # no model available: keep ml_info zeros (keywords will be applied below)
            pass
    except Exception:
        # on any model error, fallback to zeros (keywords will still apply)
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

    # NEW: ensure at least one category appears so dashboard never shows "No categories"
    if not any(ml_info.values()):
        # prefer 'related' if present, otherwise pick first category as a fallback
        if "related" in ml_info:
            ml_info["related"] = 1
        else:
            first_cat = category_names[0] if category_names else None
            if first_cat:
                ml_info[first_cat] = 1

    # If ml_info suggests severe categories, raise emergency flag
    if any(ml_info.get(c) for c in ("medical_help", "request", "aid_related", "fire")):
        is_emergency = 1

    # 3) save report to DB (ensure ml_info and assigned_helpers columns exist)
    assigned_helpers = []
    created_at = datetime.utcnow().isoformat() + "Z"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        # insert with ml_info and assigned_helpers columns; these were ensured above
        cur.execute("""
            INSERT INTO reports (message, raw_lat, raw_lon, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, ml_info, assigned_helpers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message, raw_lat, raw_lon, resolved_lat, resolved_lon, location_text, urgency, is_emergency, created_at, json.dumps(ml_info), json.dumps(assigned_helpers)))
        report_id = cur.lastrowid
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"status":"error","detail":"db insert failed","error":str(e)}), 500

    # 4) if emergency -> find helpers and notify
    if is_emergency:
        helpers = find_nearby_helpers(resolved_lat, resolved_lon, max_results=5, max_radius_km=15)
        for h in helpers:
            res_send = notify_helper_via_email(h, {
                "message": message,
                "short_msg": message[:140],
                "urgency": urgency,
                "resolved_lat": resolved_lat,
                "resolved_lon": resolved_lon,
                "location_text": location_text
            })
            assigned_helpers.append({'helper_id': h['id'], 'name': h['name'], 'email': h['email'], 'dist_km': h['dist_km'], 'send_result': res_send})
        # write assigned helpers to DB
        try:
            cur.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(assigned_helpers), report_id))
            conn.commit()
        except Exception:
            pass

    conn.close()

    return jsonify({
        "status": "ok",
        "report_id": report_id,
        "is_emergency": is_emergency,
        "urgency": urgency,
        "resolved_lat": resolved_lat,
        "resolved_lon": resolved_lon,
        "location_text": location_text,
        "assigned_helpers": assigned_helpers,
        "ml_info": ml_info
    })

@app.route('/alerts/send', methods=['POST'])
def api_send_alert():
    """
    POST JSON:
    { "to": "recipient@example.com", "subject": "text", "body": "text" }
    This handler sends the email (via src.alerts.send_email) and appends an audit
    record (one JSON object per line) to logs/alerts.log
    """
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
        # don't fail the request because of logging; include log error in response
        result = {"status": "error", "detail": f"email send result: {result}; log error: {str(e)}"}

    return jsonify(result)


@app.route('/alerts/logs', methods=['GET'])
def api_get_alert_logs():
    """
    Return recent alert audit records as JSON array.
    Query param: n (optional, default 20)
    """
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


# ---------------------------
# New helper endpoints added
# ---------------------------

@app.route('/reports/recent', methods=['GET'])
def api_get_recent_reports():
    """Return recent reports (for helper/admin dashboard). Query param: n (default 25)."""
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

# --- New endpoints: resolve, assign, export CSV ---

@app.route('/report/<int:report_id>/resolve', methods=['POST'])
def api_resolve_report(report_id):
    """Mark a report as resolved. Sets 'resolved' flag and resolved_at timestamp."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        resolved_at = datetime.utcnow().isoformat() + "Z"
        # ensure columns exist
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
    """
    Manually assign a helper to a report.
    POST JSON: { helper_id: <int> }
    """
    payload = request.get_json() or {}
    helper_id = payload.get('helper_id')
    if helper_id is None:
        return jsonify({"status":"error","detail":"missing helper_id"}), 400
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # get helper record
        cur.execute("SELECT id, name, email, lat, lon FROM helpers WHERE id = ?", (helper_id,))
        h = cur.fetchone()
        if not h:
            conn.close()
            return jsonify({"status":"error","detail":"helper not found"}), 404
        hid, name, email, hlat, hlon = h
        # fetch existing assigned_helpers
        cur.execute("SELECT assigned_helpers FROM reports WHERE id = ?", (report_id,))
        row = cur.fetchone()
        existing = []
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except Exception:
                existing = []
        # compute distance if report has coords
        cur.execute("SELECT resolved_lat, resolved_lon FROM reports WHERE id = ?", (report_id,))
        rr = cur.fetchone()
        dist_km = None
        if rr and rr[0] is not None and rr[1] is not None:
            try:
                dist_km = haversine(float(rr[0]), float(rr[1]), float(hlat), float(hlon))
                dist_km = round(dist_km, 2)
            except Exception:
                dist_km = None
        new = {"helper_id": hid, "name": name, "email": email, "dist_km": dist_km}
        existing.append(new)
        cur.execute("UPDATE reports SET assigned_helpers = ? WHERE id = ?", (json.dumps(existing), report_id))
        conn.commit()
        conn.close()
        return jsonify({"status":"ok","report_id":report_id,"assigned": new})
    except Exception as e:
        return jsonify({"status":"error","detail":str(e)}), 500

@app.route('/reports/export', methods=['GET'])
def api_export_reports_csv():
    """Export recent reports as CSV. Query param n (default 200)."""
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
    # build CSV
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id','message','location_text','resolved_lat','resolved_lon','urgency','is_emergency','created_at','assigned_helpers','ml_info'])
    for r in rows:
        writer.writerow([r[0], r[1] or '', r[4] or '', r[2] or '', r[3] or '', r[5] or '', r[6] or 0, r[7] or '', r[8] or '', r[9] or ''])
    csv_str = buf.getvalue()
    # return as text/csv response
    from flask import Response
    return Response(csv_str, mimetype='text/csv', headers={"Content-disposition":"attachment; filename=reports_export.csv"})

# ---------------------------

@app.route('/dashboard')
def dashboard():
    """
    Volunteer/Admin dashboard to view recent reports.
    Builds a 'predicted_categories' list for each report before rendering so the template
    doesn't need to use Jinja extensions.
    """
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

    # process assigned_helpers JSON and ml_info
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

        # build a predictable list of predicted categories (human-friendly order)
        preds = []
        try:
            # keep order consistent using category_names when possible
            if ml_info:
                for cname in category_names:
                    if ml_info.get(cname):
                        preds.append(cname)
            # include any keys in ml_info not in category_names
            for cname, val in ml_info.items():
                if val and cname not in preds:
                    preds.append(cname)
        except Exception:
            preds = []

        # if empty, mark as no data / none predicted
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
            "created_at": r[9] if len(r) > 9 else "N/A"
        })

    return render_template("dashboard.html", reports=reports)

@app.route('/helpers', methods=['POST'])
def api_add_helper():
    """
    Register a helper/volunteer.
    POST JSON: { name, email, phone, lat, lon, radius_km (optional, default 10) }
    """
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


# ---------------------------
# Small citizen-facing page (simple report form)
# ---------------------------
@app.route('/report')
def report_page():
    """
    Render citizen-facing report form, optionally prefilled with ?prefill=<msg>.
    """
    prefill = request.args.get("prefill", "")
    return render_template('report.html', prefill=prefill)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
