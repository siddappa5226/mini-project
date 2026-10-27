from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from image_processing import preprocess_image, extract_features

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "dataset" / "images"
LABELS_FILE = ROOT / "dataset" / "labels.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "damage_model.joblib"

def main():
    if not LABELS_FILE.exists():
        print("Missing dataset/labels.csv. Copy labels_template.csv to labels.csv and add expert-provided labels.")
        sys.exit(1)

    labels = pd.read_csv(LABELS_FILE)
    required = {"filename", "grade"}
    if not required.issubset(labels.columns):
        print("labels.csv must have columns: filename, grade")
        sys.exit(1)
    labels = labels.dropna(subset=["filename", "grade"])
    labels["filename"] = labels["filename"].astype(str)
    labels["grade"] = labels["grade"].astype(str)

    rows, y, skipped = [], [], []
    for _, item in labels.iterrows():
        path = IMAGE_DIR / item["filename"]
        if not path.is_file():
            skipped.append(item["filename"])
            continue
        try:
            image = np.array(Image.open(path).convert("RGB"))
            _, gray = preprocess_image(image)
            rows.append(extract_features(gray))
            y.append(item["grade"])
        except Exception as exc:
            print(f"Skipping {item['filename']}: {exc}")

    if skipped:
        print("Files listed in labels.csv but not found:", ", ".join(skipped))
    if len(rows) < 4:
        print(f"Only {len(rows)} usable images. Add more labelled images before training.")
        sys.exit(1)
    if len(set(y)) < 2:
        print("At least two distinct expert-provided grade classes are required.")
        sys.exit(1)

    X = pd.DataFrame(rows)
    y = pd.Series(y, name="grade")
    class_counts = y.value_counts()
    print("Images per class:")
    print(class_counts.to_string())

    # Use a holdout split only when every class has at least 2 samples.
    if class_counts.min() >= 2 and len(y) >= 8:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"Holdout accuracy (small research dataset; interpret cautiously): {accuracy_score(y_test, preds):.3f}")
        print(classification_report(y_test, preds, zero_division=0))
    else:
        print("WARNING: Too few examples per class for a reliable holdout evaluation. Training on all available images.")
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print("This is a prototype. Validate on independent, expert-labelled data before making any scientific claims.")

if __name__ == "__main__":
    main()
