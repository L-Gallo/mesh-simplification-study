"""
Pairwise comparison backend for the perceptual quality study.

Two-level balanced sampling: picks the least-shown group (model x reduction),
then the least-shown pair within that group. Each pair is shown at two viewing
distances (distant, then close). File locking (fcntl) prevents data corruption
under concurrent access on PythonAnywhere.

Author: Lukas Gallo
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
try:
    import fcntl  # POSIX file locking (Linux/Mac)
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows: no file locking, fine for local testing
import time
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

# CONFIGURATION

STUDY_CONFIG = {
    "models": ["AK74", "bunker", "church", "jeep", "LAV", "M9_pistol", "Mi8", "watermill"],
    "reduction_levels": ["50", "80", "90"],
    "methods": ["meshoptimizer", "cgal", "open3d", "fast-simplification"],

    # Session limit - max PAIRS per participant session (each pair shown twice: distant + close)
    "max_pairs_per_session": 10,  # 10 pairs = 20 total comparisons

    # View distance scaling based on reduction level
    # Higher reduction = farther distance = smaller screen size
    "view_scaling": {
        "90": 0.08,  # 90% reduced (10% tris remaining) -> 8% of frame (very far, LOD3)
        "80": 0.15,  # 80% reduced (20% tris remaining) -> 15% of frame (far, LOD2)
        "50": 0.25   # 50% reduced (50% tris remaining) -> 25% of frame (medium, LOD1)
    },

    # Image URL base. Change this when deploying.
    "image_base_url": "http://localhost:5000/static/images",
    "image_extension": ".png"

}

# File paths for data storage
DATA_FILE = "comparison_counts.json"
RESPONSES_FILE = "participant_responses.json"

# DATA STRUCTURES

def initialize_data():
    """Initialize tracking structures for groups and pairs"""
    data = {
        "groups": {},      # Tracks how many times each group has been shown
        "pairs": {},       # Tracks how many times each pair has been shown
        "metadata": {
            "initialized": datetime.now().isoformat(),
            "total_participants": 0
        }
    }

    # Create all possible groups (model x reduction_level combinations)
    for model in STUDY_CONFIG["models"]:
        for level in STUDY_CONFIG["reduction_levels"]:
            group_id = f"{model}_{level}"
            data["groups"][group_id] = {
                "count": 0,
                "model": model,
                "level": level
            }

            # Create all pairs within this group (method comparisons)
            methods = STUDY_CONFIG["methods"]
            for i, method_a in enumerate(methods):
                for method_b in methods[i+1:]:
                    pair_id = f"{group_id}_{method_a}_vs_{method_b}"
                    data["pairs"][pair_id] = {
                        "count": 0,
                        "group_id": group_id,
                        "method_a": method_a,
                        "method_b": method_b,
                        "model": model,
                        "level": level
                    }

    return data

def load_data():
    """Load existing data or create new"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return initialize_data()

def save_data(data):
    """Save data to disk. Uses file locking if available (POSIX)."""
    max_retries = 50  # 50 retries x 100ms = 5 second timeout
    retry_delay = 0.1  # 100ms between retries
    
    for attempt in range(max_retries):
        try:
            with open(DATA_FILE, 'w') as f:
                # Try to acquire exclusive lock (non-blocking)
                try:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"Lock timeout after {max_retries * retry_delay}s")
                
                try:
                    json.dump(data, f, indent=2)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                return  # Success
                
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                print(f"ERROR saving data: {e}")
                raise
            time.sleep(retry_delay)

def load_responses():
    """Load participant responses"""
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'r') as f:
            return json.load(f)
    return {"responses": []}

def save_response(response_data):
    """Append a response. Uses file locking if available (POSIX)."""
    max_retries = 50  # 50 retries x 100ms = 5 second timeout
    retry_delay = 0.1  # 100ms between retries
    
    for attempt in range(max_retries):
        try:
            # Read-modify-write with exclusive lock
            with open(RESPONSES_FILE, 'r+' if os.path.exists(RESPONSES_FILE) else 'w+') as f:
                # Try to acquire exclusive lock (non-blocking)
                try:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    # Lock not available - another process is writing
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"Timeout acquiring lock after {max_retries * retry_delay}s")
                
                try:
                    # Read existing responses
                    f.seek(0)
                    try:
                        responses = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        # Empty or corrupted file - initialize
                        responses = {"responses": []}
                    
                    # Append new response
                    responses["responses"].append(response_data)
                    
                    # Write back
                    f.seek(0)
                    json.dump(responses, f, indent=2)
                    f.truncate()  # Remove any leftover data if file shrank
                    
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                return  # Success
                
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                print(f"ERROR saving response: {e}")
                raise
            time.sleep(retry_delay)

# ADAPTIVE SAMPLING LOGIC

def get_image_url(model, level, method):
    """Generate image URL from naming convention: model_method_level.png"""
    filename = f"{model}_{method}_{level}{STUDY_CONFIG['image_extension']}"
    return f"{STUDY_CONFIG['image_base_url']}/{filename}"

def select_next_pair(participant_id, session_data):
    """
    Adaptive selection algorithm for paired distant/close comparisons:
    1. Find groups with lowest presentation count
    2. Among those groups, find pair with lowest presentation count
    3. Return pair that hasn't been shown to this participant yet
    4. Determine which view to show next (distant or close)
    5. Enforce session limit (max pairs, not comparisons)
    """
    data = load_data()

    # Track what this participant has already seen
    if participant_id not in session_data:
        session_data[participant_id] = {
            "shown_pairs": {},  # Dict: pair_id -> {"distant": bool, "close": bool, "order": str}
            "pairs_completed": 0,  # Number of fully completed pairs (both views)
            "start_time": datetime.now().isoformat()
        }

    session = session_data[participant_id]

    # Check session limit (based on PAIRS, not individual comparisons)
    max_pairs = STUDY_CONFIG.get("max_pairs_per_session", 10)
    if session["pairs_completed"] >= max_pairs:
        return None  # Session limit reached

    # Find pairs that need views shown
    available_pairs = {}

    for pair_id, pair_info in data["pairs"].items():
        if pair_id not in session["shown_pairs"]:
            # Brand new pair - add to available
            available_pairs[pair_id] = pair_info
        elif not session["shown_pairs"][pair_id].get("distant") or not session["shown_pairs"][pair_id].get("close"):
            # Pair partially shown - needs second view
            available_pairs[pair_id] = pair_info

    if not available_pairs:
        return None  # Participant has seen all pairs (both views)

    # Separate pairs into: new pairs vs pairs needing second view
    new_pairs = {pid: info for pid, info in available_pairs.items()
                 if pid not in session["shown_pairs"]}
    partial_pairs = {pid: info for pid, info in available_pairs.items()
                     if pid in session["shown_pairs"]}

    # Prioritize showing second view of partially-shown pairs
    if partial_pairs:
        # Must show second view of a partial pair
        selected_pair_id = list(partial_pairs.keys())[0]  # Just get the first one
        pair_info = partial_pairs[selected_pair_id]

        # Determine which view hasn't been shown yet
        if not session["shown_pairs"][selected_pair_id].get("distant"):
            view_type = "distant"
        else:
            view_type = "close"
    else:
        # Select new pair using adaptive sampling
        # Step 1: Find groups with minimum count
        group_counts = {
            group_id: info["count"]
            for group_id, info in data["groups"].items()
        }
        min_group_count = min(group_counts.values())
        low_exposure_groups = [
            group_id for group_id, count in group_counts.items()
            if count == min_group_count
        ]

        # Step 2: Filter pairs from low-exposure groups
        candidate_pairs = {
            pair_id: info for pair_id, info in new_pairs.items()
            if info["group_id"] in low_exposure_groups
        }

        if not candidate_pairs:
            candidate_pairs = new_pairs

        # Step 3: Among candidates, find pair with minimum count
        min_pair_count = min(info["count"] for info in candidate_pairs.values())
        lowest_pairs = [
            pair_id for pair_id, info in candidate_pairs.items()
            if info["count"] == min_pair_count
        ]

        # Step 4: Randomly select from ties
        selected_pair_id = random.choice(lowest_pairs)
        pair_info = data["pairs"][selected_pair_id]

        # Randomize which view comes first (50/50)
        view_type = random.choice(["distant", "close"])

        # Initialize tracking for this pair
        session["shown_pairs"][selected_pair_id] = {
            "distant": False,
            "close": False,
            "order": f"{view_type}_first"
        }

    # Mark this view as shown
    session["shown_pairs"][selected_pair_id][view_type] = True

    # Check if pair is now complete (both views shown)
    if session["shown_pairs"][selected_pair_id]["distant"] and session["shown_pairs"][selected_pair_id]["close"]:
        session["pairs_completed"] += 1

    # Randomize left/right position (A/B order) - INDEPENDENT for each view
    methods = [pair_info["method_a"], pair_info["method_b"]]
    random.shuffle(methods)

    # Get scaling factor for this reduction level
    scale_factor = STUDY_CONFIG["view_scaling"].get(pair_info["level"], 0.25)

    return {
        "pair_id": selected_pair_id,
        "view_type": view_type,
        "scale_factor": scale_factor,
        "group_id": pair_info["group_id"],
        "model": pair_info["model"],
        "level": pair_info["level"],
        "method_left": methods[0],
        "method_right": methods[1],
        "image_left": get_image_url(pair_info["model"], pair_info["level"], methods[0]),
        "image_right": get_image_url(pair_info["model"], pair_info["level"], methods[1]),
        "current_pair_count": pair_info["count"],
        "current_group_count": data["groups"][pair_info["group_id"]]["count"],
        "order_in_pair": 1 if view_type == session["shown_pairs"][selected_pair_id]["order"].split("_")[0] else 2
    }

# API ENDPOINTS

# In-memory session storage (resets on server restart - acceptable for thesis)
SESSION_DATA = {}

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Mesh Comparison Backend API",
        "endpoints": [
            "/next_pair",
            "/submit_response",
            "/stats",
            "/leaderboard",
            "/method_stats"
        ]
    })

@app.route('/next_pair', methods=['GET'])
def next_pair():
    """
    Get next comparison for participant (distant or close view)
    Each pair is shown twice: once distant, once close
    Query params: participant_id (optional, generated if not provided)
    """
    participant_id = request.args.get('participant_id', f"p_{datetime.now().timestamp()}")

    try:
        pair_data = select_next_pair(participant_id, SESSION_DATA)

        if pair_data is None:
            # Check if session limit reached or all pairs complete
            if participant_id in SESSION_DATA:
                pairs_completed = SESSION_DATA[participant_id]["pairs_completed"]
                max_pairs = STUDY_CONFIG.get("max_pairs_per_session", 10)

                if pairs_completed >= max_pairs:
                    # Session limit reached
                    return jsonify({
                        "status": "session_complete",
                        "message": "Session limit reached",
                        "pairs_completed": pairs_completed,
                        "total_comparisons": pairs_completed * 2,
                        "session_limit": max_pairs
                    })
                else:
                    # All pairs complete
                    return jsonify({
                        "status": "complete",
                        "message": "All pairs completed"
                    })
            else:
                return jsonify({
                    "status": "complete",
                    "message": "All pairs completed"
                })

        # Calculate total comparisons completed
        shown_pairs = SESSION_DATA[participant_id]["shown_pairs"]
        total_comparisons = sum(
            (1 if p.get("distant") else 0) + (1 if p.get("close") else 0)
            for p in shown_pairs.values()
        )

        return jsonify({
            "status": "success",
            "participant_id": participant_id,
            "comparison": pair_data,
            "progress": {
                "comparisons_completed": total_comparisons,
                "total_comparisons": STUDY_CONFIG.get("max_pairs_per_session", 10) * 2,
                "pairs_completed": SESSION_DATA[participant_id]["pairs_completed"],
                "total_pairs": STUDY_CONFIG.get("max_pairs_per_session", 10)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/submit_response', methods=['POST'])
def submit_response():
    """
    Submit participant choice for one view (distant or close)
    Body: {
        "participant_id": "...",
        "pair_id": "...",
        "view_type": "distant" or "close",
        "chosen_side": "left" or "right",
        "chosen_method": "...",
        "not_chosen_method": "...",
        "reaction_time_ms": 1234,
        "order_in_pair": 1 or 2
    }
    """
    try:
        response_data = request.json

        # Validate required fields
        required = ["participant_id", "pair_id", "view_type", "chosen_side", "chosen_method", "not_chosen_method"]
        if not all(field in response_data for field in required):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Validate view_type
        if response_data["view_type"] not in ["distant", "close"]:
            return jsonify({"status": "error", "message": "Invalid view_type"}), 400

        # Update counts only when BOTH views of a pair are complete
        data = load_data()
        pair_id = response_data["pair_id"]

        if pair_id not in data["pairs"]:
            return jsonify({"status": "error", "message": "Invalid pair_id"}), 400

        # Check if this completes the pair (both views now shown)
        participant_id = response_data["participant_id"]
        if participant_id in SESSION_DATA:
            pair_status = SESSION_DATA[participant_id]["shown_pairs"].get(pair_id, {})
            both_views_complete = pair_status.get("distant", False) and pair_status.get("close", False)

            # Only increment counts when BOTH views are complete
            if both_views_complete:
                data["pairs"][pair_id]["count"] += 1
                group_id = data["pairs"][pair_id]["group_id"]
                data["groups"][group_id]["count"] += 1
                save_data(data)

        # Save response with view_type information
        response_data["timestamp"] = datetime.now().isoformat()
        save_response(response_data)

        return jsonify({
            "status": "success",
            "message": "Response recorded",
            "view_type": response_data["view_type"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get current statistics"""
    data = load_data()
    responses = load_responses()

    return jsonify({
        "total_groups": len(data["groups"]),
        "total_pairs": len(data["pairs"]),
        "total_responses": len(responses["responses"]),
        "unique_participants": len(SESSION_DATA),
        "group_counts": {
            group_id: info["count"]
            for group_id, info in sorted(data["groups"].items())
        },
        "pair_counts": {
            pair_id: info["count"]
            for pair_id, info in sorted(data["pairs"].items())
        }
    })

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    """
    Get method rankings based on win counts
    Optional query param: view_type (distant/close/all)
    Returns sorted list of methods by number of times they were chosen
    """
    view_filter = request.args.get('view_type', 'all')  # 'distant', 'close', or 'all'

    responses = load_responses()

    # Filter responses by view type if specified
    filtered_responses = responses["responses"]
    if view_filter in ['distant', 'close']:
        filtered_responses = [r for r in responses["responses"] if r.get("view_type") == view_filter]

    # Count wins per method
    method_wins = {method: 0 for method in STUDY_CONFIG["methods"]}
    method_comparisons = {method: 0 for method in STUDY_CONFIG["methods"]}

    for response in filtered_responses:
        chosen = response["chosen_method"]
        not_chosen = response.get("not_chosen_method")

        if chosen in method_wins:
            method_wins[chosen] += 1

        # Count total comparisons (both chosen and not chosen participated)
        if chosen in method_comparisons:
            method_comparisons[chosen] += 1
        if not_chosen and not_chosen in method_comparisons:
            method_comparisons[not_chosen] += 1

    # Calculate win rates
    leaderboard_data = []
    for method in STUDY_CONFIG["methods"]:
        wins = method_wins[method]
        total = method_comparisons[method]
        win_rate = (wins / total * 100) if total > 0 else 0

        leaderboard_data.append({
            "method": method,
            "wins": wins,
            "total_comparisons": total,
            "losses": total - wins,
            "win_rate": round(win_rate, 1)
        })

    # Sort by wins (descending)
    leaderboard_data.sort(key=lambda x: x["wins"], reverse=True)

    # Add rankings
    for i, item in enumerate(leaderboard_data, 1):
        item["rank"] = i

    return jsonify({
        "view_type": view_filter,
        "total_responses": len(filtered_responses),
        "leaderboard": leaderboard_data
    })

@app.route('/method_stats', methods=['GET'])
def method_stats():
    """
    Detailed statistics per method including head-to-head records
    """
    responses = load_responses()

    # Initialize tracking
    method_stats = {
        method: {
            "total_wins": 0,
            "total_comparisons": 0,
            "wins_by_model": {},
            "wins_by_level": {},
            "head_to_head": {other: {"wins": 0, "losses": 0}
                             for other in STUDY_CONFIG["methods"] if other != method}
        }
        for method in STUDY_CONFIG["methods"]
    }

    # Process responses
    for response in responses["responses"]:
        pair_id = response["pair_id"]
        chosen = response["chosen_method"]

        # Parse pair_id to get both methods and context
        # Format: model_level_methodA_vs_methodB
        parts = pair_id.split('_')

        # Find 'vs' to split methods
        vs_index = parts.index('vs')
        method_a = parts[vs_index - 1]
        method_b = parts[vs_index + 1]

        # Get model and level (everything before last method)
        model = parts[0]
        level = parts[1]

        # Track wins
        if chosen in method_stats:
            method_stats[chosen]["total_wins"] += 1

            # Track by model
            if model not in method_stats[chosen]["wins_by_model"]:
                method_stats[chosen]["wins_by_model"][model] = 0
            method_stats[chosen]["wins_by_model"][model] += 1

            # Track by level
            if level not in method_stats[chosen]["wins_by_level"]:
                method_stats[chosen]["wins_by_level"][level] = 0
            method_stats[chosen]["wins_by_level"][level] += 1

        # Track head-to-head
        if chosen == method_a:
            method_stats[method_a]["head_to_head"][method_b]["wins"] += 1
            method_stats[method_b]["head_to_head"][method_a]["losses"] += 1
        elif chosen == method_b:
            method_stats[method_b]["head_to_head"][method_a]["wins"] += 1
            method_stats[method_a]["head_to_head"][method_b]["losses"] += 1

    # Count total comparisons from pair counts
    data = load_data()
    for pair_id, pair_info in data["pairs"].items():
        count = pair_info["count"]
        method_stats[pair_info["method_a"]]["total_comparisons"] += count
        method_stats[pair_info["method_b"]]["total_comparisons"] += count

    # Calculate win rates
    for method in method_stats:
        total = method_stats[method]["total_comparisons"]
        wins = method_stats[method]["total_wins"]
        method_stats[method]["win_rate"] = round((wins / total * 100) if total > 0 else 0, 1)
        method_stats[method]["loss_rate"] = round(((total - wins) / total * 100) if total > 0 else 0, 1)

    return jsonify({
        "total_responses": len(responses["responses"]),
        "methods": method_stats
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset all data (use with caution!)"""
    password = request.json.get('password')
    if password != "reset_my_data_2024":  # Change this password!
        return jsonify({"status": "error", "message": "Invalid password"}), 403

    data = initialize_data()
    save_data(data)

    # Clear responses
    with open(RESPONSES_FILE, 'w') as f:
        json.dump({"responses": []}, f)

    SESSION_DATA.clear()

    return jsonify({"status": "success", "message": "All data reset"})

# MAIN

if __name__ == '__main__':
    # Initialize data files if they don't exist
    if not os.path.exists(DATA_FILE):
        save_data(initialize_data())

    # Run in debug mode for local testing
    # On PythonAnywhere, this runs via WSGI automatically
    app.run(debug=True, host='0.0.0.0', port=5000)