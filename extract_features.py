
from pathlib import Path
import cv2
import csv
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "dataset" / "images"

ANNOTATION_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

OUTPUT_FILE = PROJECT_DIR / "corneal_image_features.csv"


def count_annotated_points(annotation_path):
    """Count valid annotation rows containing x and y coordinates."""
    if not annotation_path.exists():
        return None

    count = 0

    with annotation_path.open("r", encoding="utf-8-sig") as file:
        next(file, None)  # Skip the header

        for line in file:
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                int(parts[1])  # x coordinate
                int(parts[2])  # y coordinate
                count += 1
            except ValueError:
                continue

    return count


def main():
    if not IMAGE_DIR.is_dir():
        print(f"Image folder not found: {IMAGE_DIR}")
        return

    if not ANNOTATION_DIR.is_dir():
        print(f"Annotation folder not found: {ANNOTATION_DIR}")
        return

    rows = []
    skipped = 0

    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    image_files = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    for image_path in image_files:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read image: {image_path.name}")
            skipped += 1
            continue

        height, width = image.shape

        mean_intensity = float(np.mean(image))
        contrast = float(np.std(image))

        annotation_path = ANNOTATION_DIR / (
            image_path.stem + ".txt"
        )

        cell_count = count_annotated_points(annotation_path)

        if cell_count is None:
            print(f"Missing annotation: {annotation_path.name}")

        rows.append({
            "image": image_path.name,
            "width": width,
            "height": height,
            "mean_intensity": round(mean_intensity, 2),
            "contrast": round(contrast, 2),
            "annotated_cell_points": (
                cell_count if cell_count is not None else ""
            )
        })

    if not rows:
        print("No images were processed.")
        return

    columns = [
        "image",
        "width",
        "height",
        "mean_intensity",
        "contrast",
        "annotated_cell_points"
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFEATURE EXTRACTION COMPLETED")
    print("----------------------------")
    print(f"Images found: {len(image_files)}")
    print(f"Images processed: {len(rows)}")
    print(f"Images skipped: {skipped}")
    print(f"CSV saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
