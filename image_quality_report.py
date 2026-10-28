
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

ROOT = Path(__file__).parent
IMAGE_DIR = Path(r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels")
OUTPUT_FILE = ROOT / "image_quality_report.csv"

rows = []

for path in sorted(IMAGE_DIR.iterdir()):
    if path.suffix.lower() not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        continue

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not read:", path.name)
        continue

    rows.append({
        "image": path.name,
        "width": image.shape[1],
        "height": image.shape[0],
        "mean_brightness": round(float(np.mean(image)), 2),
        "contrast": round(float(np.std(image)), 2),
        "sharpness": round(
            float(cv2.Laplacian(image, cv2.CV_64F).var()), 2
        )
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("Images analysed:", len(df))
print("Report saved to:", OUTPUT_FILE)
print("\nSummary:")
print(df[["mean_brightness", "contrast", "sharpness"]].describe().round(2))

if not df.empty:
    df[["mean_brightness", "contrast", "sharpness"]].hist(
        bins=20, figsize=(10, 6)
    )
    plt.tight_layout()
    plt.show()
