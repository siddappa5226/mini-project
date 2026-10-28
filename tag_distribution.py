
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

tag_counts = Counter()
images_with_tag = Counter()

for path in DATASET_DIR.glob("*.txt"):
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()[1:]  # Skip header

    image_tags = set()

    for line in lines:
        parts = line.split()

        if len(parts) >= 4:
            tag = parts[3]
            tag_counts[tag] += 1
            image_tags.add(tag)

    for tag in image_tags:
        images_with_tag[tag] += 1

print("Total annotation points by tag:")
for tag, count in sorted(tag_counts.items()):
    print(f"Tag {tag}: {count} points")

print("\nImages containing each tag:")
for tag, count in sorted(images_with_tag.items()):
    print(f"Tag {tag}: {count} images")
