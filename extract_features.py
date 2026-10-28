
import os
import cv2
import csv
import numpy as np

folder = r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
output_file = "corneal_image_features.csv"

rows = []

for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    image_path = os.path.join(folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not read:", filename)
        continue

    height, width = image.shape
    mean_intensity = float(np.mean(image))
    contrast = float(np.std(image))

    annotation_path = os.path.join(
        folder, os.path.splitext(filename)[0] + ".txt"
    )

    cell_count = 0

    if os.path.exists(annotation_path):
        with open(annotation_path, "r", encoding="utf-8") as file:
            for line in file.readlines()[1:]:
                parts = line.split()

                if len(parts) >= 4:
                    try:
                        int(parts[1])
                        int(parts[2])
                        cell_count += 1
                    except ValueError:
                        pass

    rows.append({
        "image": filename,
        "width": width,
        "height": height,
        "mean_intensity": round(mean_intensity, 2),
        "contrast": round(contrast, 2),
        "annotated_cell_points": cell_count
    })

if rows:
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print("Feature extraction completed!")
print("Images processed:", len(rows))
print("CSV saved:", os.path.abspath(output_file))
