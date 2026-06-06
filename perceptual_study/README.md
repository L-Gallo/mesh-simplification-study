# Perceptual Quality Study

Flask-based pairwise comparison tool used to collect perceptual quality
judgments for the thesis. Participants see two renders of the same mesh
simplified by different methods and choose which looks better. Each pair
is shown at two viewing distances (distant and close, order randomized)
to simulate in-game LOD conditions. The backend uses adaptive sampling
to balance exposure across all method pairs and reduction levels.

Deployed on PythonAnywhere for the thesis study. Collected ~2800 judgments
from ~150 participants. The anonymized response data is in
`data/participant_responses.json` at the repo root.

## Files

**app.py** -- Flask backend. Serves pairs via `/next_pair`, records choices
via `/submit_response`. Adaptive sampling picks the least-shown group
(model x reduction) first, then the least-shown pair within that group.
Uses fcntl file locking for concurrent writes (POSIX only, skipped on
Windows for local testing).

**test_interface.html** -- Participant-facing page. Shows two renders side
by side, first at a scaled-down size (distant view) then at full size
(close view). Records the choice and reaction time for each view.
Standalone HTML that calls the backend API via fetch.

**admin_interface.html** -- Live dashboard showing response counts, win
rates, and head-to-head records. Used during data collection to monitor
progress.

**verify_images.py** -- Checks that all required render images exist.
Config must match `STUDY_CONFIG` in `app.py`. Pass an image directory
as argument or it defaults to `./images/`.

## Setup

Render images are named `{model}_{method}_{level}.png` (e.g.
`church_cgal_80.png`). Generate them from your simplified meshes
and place them in an images directory.

```bash
pip install flask flask-cors

# Check all images are present
python verify_images.py ./path/to/images/

# Run locally
python app.py
# Backend at http://localhost:5000
# Open test_interface.html in a browser
```

Edit `STUDY_CONFIG` in `app.py` to set models, methods, reduction levels,
view scaling factors, image URL, and session length. Update `API_BASE` in
`test_interface.html` to match your deployment URL.

## Deployment (PythonAnywhere)

1. Upload `app.py`, both HTML files, and your render images.
2. Set `image_base_url` in `STUDY_CONFIG` to your static URL.
3. Update `API_BASE` in `test_interface.html` to your app URL.
4. Point the WSGI config to `app.py`.
5. Share `test_interface.html` URL with participants.
6. Monitor via `admin_interface.html`.

## Output

The app generates two JSON files at runtime (not committed to the repo):

`comparison_counts.json` -- tracks how many times each pair and group has
been shown. Used by the adaptive sampler.

`participant_responses.json` -- one entry per judgment with participant ID,
pair ID, view type (distant/close), chosen side/method, reaction time,
and timestamp.