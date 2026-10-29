
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "dataset" / "images"

ANNOTATION_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

image_path = IMAGE_DIR / "71866.tiff"
annotation_path = ANNOTATION_DIR / "71866.txt"

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

if not annotation_path.exists():
    raise FileNotFoundError(f"Cannot find annotation: {annotation_path}")

plt.figure(figsize=(8, 10))
plt.imshow(image, cmap="gray")

with annotation_path.open("r", encoding="utf-8-sig") as file:
    next(file, None)  # Skip the header

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

plt.title("Corneal Endothelial Image with Annotated Points")
plt.axis("off")
plt.tight_layout()
plt.show()
