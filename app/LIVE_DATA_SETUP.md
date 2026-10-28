# DisasterSense AI — Live Maharashtra Data (first version)

This update adds a live environmental monitoring section to `app/templates/dashboard.html` and a `/api/live-data` endpoint to `app/run.py`. It preserves the existing prediction, message classification, emergency report and rescue-tracking routes.

## Data feeds
- **Open-Meteo**: current conditions and 24-hour hourly forecast for Mumbai, Pune, Nashik, Nagpur and Chhatrapati Sambhajinagar. Current weather and forecast are labelled separately; hourly precipitation chart is forecast, not observed rain. Open-Meteo is called server-side and cached for 4 minutes; browser refresh target is 5 minutes.
- **NASA FIRMS**: latest VIIRS NOAA-20 NRT hotspots within an approximate Maharashtra bounding box. Requires a free MAP_KEY. If missing or rejected, the UI shows that the feed is not connected rather than fake points.
- **NASA GPM IMERG via NASA GIBS**: precipitation-rate imagery requested as a WMS map overlay. This first version does not calculate point-level rainfall time series from GPM raster pixels. The chart uses Open-Meteo hourly forecast values and is explicitly labelled forecast. The GIBS overlay may not be available for every requested date or layer.

## Setup
1. Back up the existing project and templates.
2. Copy the included `app/run.py` and `app/templates/dashboard.html` into the matching locations. Keep all other project files and folders.
3. Copy `.env.example` to `.env` in the project root. Add `FIRMS_MAP_KEY` after requesting a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key. Do not commit `.env`.
4. Install existing project dependencies as usual. No new Python package is required; browser charts use Chart.js and maps use Leaflet via CDN.
5. Start with `python app/run.py`, open `/dashboard`, and check `/api/live-data` directly if the dashboard reports an error.
6. Optional existing route/ETA integration uses `ORS_API_KEY` from environment. The hard-coded token was removed from `run.py`; set a replacement key in your local environment if the ETA integration requires it. Because a credential was present in the earlier source, rotate/revoke that old token.

## Important limitations
- Refresh interval is a UI/backend fetch cadence, not the satellite's capture cadence. NASA data can be delayed by processing and product availability.
- FIRMS hotspots are thermal anomalies and do not prove a confirmed fire.
- GPM layer date/time and tile availability should be checked against NASA GIBS. The overlay is imagery, not raw calibrated rainfall points.
- This package was syntax-checked locally, but not run against your full database, model files, installed Python environment, API key or network. Test existing emergency workflows after merging.
- The existing project starts on port 3001.
