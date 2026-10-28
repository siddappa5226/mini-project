
import os
from collections import Counter

folder = r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"

tag_counts = Counter()
total_files = 0
empty_files = 0

for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue

    total_files += 1
    path = os.path.join(folder, filename)

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()[1:]

    if not lines:
        empty_files += 1

    for line in lines:
        parts = line.split()

        if len(parts) >= 4:
            tag_counts[parts[3]] += 1

print("Annotation files:", total_files)
print("Empty annotations:", empty_files)
print("Tag counts:", tag_counts)
