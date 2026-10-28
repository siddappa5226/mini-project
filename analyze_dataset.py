
import os

folder = r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"

total_images = 0
total_cells = 0
empty_annotations = 0
missing_annotations = 0

for filename in os.listdir(folder):
    if filename.lower().endswith((".tif", ".tiff")):
        total_images += 1

        annotation_name = os.path.splitext(filename)[0] + ".txt"
        annotation_path = os.path.join(folder, annotation_name)

        if not os.path.exists(annotation_path):
            missing_annotations += 1
            continue

        with open(annotation_path, "r", encoding="utf-8") as file:
            lines = file.readlines()[1:]

        cell_count = 0

        for line in lines:
            parts = line.split()

            if len(parts) >= 4:
                try:
                    int(parts[1])
                    int(parts[2])
                    cell_count += 1
                except ValueError:
                    continue

        total_cells += cell_count

        if cell_count == 0:
            empty_annotations += 1

print("DATASET ANALYSIS")
print("----------------")
print("Total images:", total_images)
print("Total annotated cells:", total_cells)
print("Average annotated cells per image:",
      round(total_cells / total_images, 2) if total_images else 0)
print("Images with empty annotations:", empty_annotations)
print("Images missing annotations:", missing_annotations)
