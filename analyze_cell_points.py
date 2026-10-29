
from pathlib import Path
import csv
import math
import statistics

PROJECT_DIR = Path(__file__).resolve().parent
ANNOTATION_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

ANNOTATION_FILE = ANNOTATION_DIR / "71866.txt"


def main():
    points = []

    with ANNOTATION_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                tag = row["tag"]
                points.append((x, y, tag))
            except (ValueError, KeyError, TypeError):
                continue

    if not points:
        print("No valid annotation points found.")
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    nearest_distances = []

    for i, (x, y, _) in enumerate(points):
        distances = [
            math.hypot(x - other_x, y - other_y)
            for j, (other_x, other_y, _) in enumerate(points)
            if i != j
        ]

        if distances:
            nearest_distances.append(min(distances))

    print("CELL-POINT ANALYSIS")
    print("-------------------")
    print("Annotation points:", len(points))
    print("Unique tags:", sorted({p[2] for p in points}))
    print("X coordinate range:", min(xs), "to", max(xs))
    print("Y coordinate range:", min(ys), "to", max(ys))

    if nearest_distances:
        print(
            "Average nearest-point distance:",
            round(statistics.mean(nearest_distances), 2),
            "pixels"
        )


if __name__ == "__main__":
    main()
