"""Check that all required render images exist before deploying the study."""

import os
import sys

# Must match STUDY_CONFIG in app.py
MODELS = ["AK74", "bunker", "church", "jeep", "LAV", "M9_pistol", "Mi8", "watermill"]
REDUCTION_LEVELS = ["50", "80", "90"]
METHODS = ["meshoptimizer", "cgal", "open3d", "fast-simplification"]

IMAGE_DIR = "./images/"
if len(sys.argv) > 1:
    IMAGE_DIR = sys.argv[1]

# Build expected filenames: {model}_{method}_{level}.png
expected = []
for model in MODELS:
    for level in REDUCTION_LEVELS:
        for method in METHODS:
            expected.append(f"{model}_{method}_{level}.png")

print(f"Checking {len(expected)} images in {os.path.abspath(IMAGE_DIR)}")

missing = [f for f in expected if not os.path.exists(os.path.join(IMAGE_DIR, f))]

if missing:
    print(f"\nMissing {len(missing)} of {len(expected)}:")
    for f in missing:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"All {len(expected)} images found.")