
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from image_processing import preprocess_image, extract_features

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "dataset" / "images"
ANNOTATION_DIR = IMAGE_DIR / "images_and_labels"
OUTPUT_FILE = ROOT / "dataset" / "training_features.csv"

rows = []

for image_path in sorted(IMAGE_DIR.glob("*.tiff")):
    annotation_path = ANNOTATION_DIR / f"{image_path.stem}.txt"

    if not annotation_path.exists():
        print(f"Missing annotation: {image_path.name}")
        continue

    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        _, gray = preprocess_image(image)

        features = extract_features(gray)

        points = pd.read_csv(annotation_path, sep=r"\s+")

        if not {"x", "y"}.issubset(points.columns):
            print(f"Invalid annotation: {annotation_path.name}")
            continue

        xy = points[["x", "y"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna().to_numpy()

        height, width = gray.shape

        features["filename"] = image_path.name
        features["cell_point_count"] = len(xy)
        features["point_density_per_10000_pixels"] = (
            len(xy) / (height * width) * 10000
        )

        if len(xy) >= 2:
            distances = cdist(xy, xy)
            np.fill_diagonal(distances, np.inf)
            nearest = distances.min(axis=1)
            features["mean_nearest_point_distance"] = float(
                nearest.mean()
            )
        else:
            features["mean_nearest_point_distance"] = np.nan

        rows.append(features)

    except Exception as exc:
        print(f"Skipped {image_path.name}: {exc}")

if not rows:
    raise SystemExit("No images were processed. Check the image and annotation paths.")

data = pd.DataFrame(rows)
data.insert(1, "grade", np.nan)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_FILE, index=False)

print(f"\nImages processed: {len(data)}")
print(f"Feature file saved: {OUTPUT_FILE}")
print("Grade is blank because expert-provided labels are not available.")
print(data.head().to_string(index=False))
