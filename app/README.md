# DisasterSense AI — Dark Premium UI (layout fix)

## Changes
- Fixed the hero section background so its dark area grows with the content and text no longer overlaps the light page section.
- Improved heading and disclaimer line-height/wrapping, including mobile layouts.
- Removed the training-data graphs section and Plotly rendering from the homepage.
- Replaced the graph-count summary card with an emergency-tools summary.
- Kept prediction controls, session prediction table, classifier form, live dashboard link, emergency report page, and recent alerts UI.
- Preserved the deep navy, warm ivory, teal, amber, and alert-red palette.

## Install
1. Stop the Flask app.
2. Back up your existing `app/templates` folder.
3. Extract the ZIP into the project root `D:\\Disaster-Management`, preserving folder paths.
4. Run `python app/run.py`.
5. Hard refresh the browser with Ctrl+F5.

The package changes templates only. Backend functionality and all routes should still be tested locally. Prediction history remains browser-session-only unless the backend is separately updated to persist it.
