
from pathlib import Path
import csv
import math
import statistics

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "dataset" / "images"

ANNOTATION_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

OUTPUT_FILE = PROJECT_DIR / "cell_point_analysis.csv"


def analyze_annotation(file_path):
    points = []

    with file_path.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                points.append((x, y))
            except (ValueError, KeyError, TypeError):
                continue

    if not points:
        return {
            "annotated_points": 0,
            "average_nearest_distance": ""
        }

    nearest_distances = []

    for i, (x, y) in enumerate(points):
        nearest = min(
            math.hypot(x - other_x, y - other_y)
            for j, (other_x, other_y) in enumerate(points)
            if i != j
        ) if len(points) > 1 else None

        if nearest is not None:
            nearest_distances.append(nearest)

    average_distance = (
        round(statistics.mean(nearest_distances), 2)
        if nearest_distances else ""
    )

    return {
        "annotated_points": len(points),
        "average_nearest_distance": average_distance
    }


def main():
    rows = []

    for image_path in sorted(IMAGE_DIR.glob("*.tiff")):
        annotation_path = ANNOTATION_DIR / (
            image_path.stem + ".txt"
        )

        if not annotation_path.exists():
            rows.append({
                "image": image_path.name,
                "annotated_points": "",
                "average_nearest_distance": "",
                "status": "Missing annotation"
            })
            continue

        result = analyze_annotation(annotation_path)

        rows.append({
            "image": image_path.name,
            **result,
            "status": "OK" if result["annotated_points"] > 0
                      else "No annotated points"
        })

    columns = [
        "image",
        "annotated_points",
        "average_nearest_distance",
        "status"
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("ALL-IMAGE ANALYSIS COMPLETED")
    print("----------------------------")
    print("Images analyzed:", len(rows))
    print("Images with points:", sum(
        row["status"] == "OK" for row in rows
    ))
    print("Images with no points:", sum(
        row["status"] == "No annotated points" for row in rows
    ))
    print("Missing annotation files:", sum(
        row["status"] == "Missing annotation" for row in rows
    ))
    print("CSV saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
