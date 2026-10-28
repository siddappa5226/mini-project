
import os
import csv

folder = r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
output_file = "cell_count_report.csv"

rows = []

for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    image_id = os.path.splitext(filename)[0]
    annotation_path = os.path.join(folder, image_id + ".txt")
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
        "annotation_file": image_id + ".txt",
        "annotated_cell_points": cell_count,
        "has_annotations": cell_count > 0
    })

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Report created:", os.path.abspath(output_file))
print("Images analyzed:", len(rows))
print("Images with annotated points:", sum(r["has_annotations"] for r in rows))
print("Images with zero annotated points:", sum(not r["has_annotations"] for r in rows))
