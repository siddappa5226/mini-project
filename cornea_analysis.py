
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "dataset" / "images"
ANNOTATION_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

CSV_FILE = PROJECT_DIR / "corneal_image_features.csv"


def main():
    df = pd.read_csv(CSV_FILE)

    # Select the first image with annotated points
    available = df[df["annotated_cell_points"] > 0]

    if available.empty:
        print("No images with annotated points were found.")
        return

    filename = available.iloc[0]["image"]
    image_path = IMAGE_DIR / filename
    annotation_path = ANNOTATION_DIR / (Path(filename).stem + ".txt")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not load:", image_path)
        return

    height, width = image.shape
    mean_intensity = image.mean()
    contrast = image.std()

    plt.figure(figsize=(8, 10))
    plt.imshow(image, cmap="gray")

    point_count = 0

    if annotation_path.exists():
        with annotation_path.open("r", encoding="utf-8-sig") as file:
            next(file, None)

            for line in file:
                parts = line.split()

                if len(parts) < 4:
                    continue

                try:
                    x = int(parts[1])
                    y = int(parts[2])
                except ValueError:
                    continue

                plt.scatter(x, y, s=10, c="red", marker="+")
                point_count += 1

    plt.title(f"{filename} — Annotated points: {point_count}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nIMAGE ANALYSIS")
    print("--------------")
    print("Image:", filename)
    print("Dimensions:", width, "x", height)
    print("Mean intensity:", round(mean_intensity, 2))
    print("Contrast:", round(contrast, 2))
    print("Annotated points:", point_count)


if __name__ == "__main__":
    main()
