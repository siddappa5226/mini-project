
from pathlib import Path
import pandas as pd

DATASET_DIR = Path(
    r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels"
)

rows = []

for path in sorted(DATASET_DIR.glob("*.txt")):
    counts = {"0": 0, "1": 0}

    with path.open("r", encoding="utf-8") as file:
        next(file, None)  # Skip header

        for line in file:
            parts = line.split()

            if len(parts) >= 4 and parts[3] in counts:
                counts[parts[3]] += 1

    rows.append({
        "image": path.stem + ".tiff",
        "tag_0_points": counts["0"],
        "tag_1_points": counts["1"],
        "total_points": counts["0"] + counts["1"]
    })

df = pd.DataFrame(rows)

output = Path(__file__).parent / "tag_count_per_image.csv"
df.to_csv(output, index=False)

print("Annotation files analysed:", len(df))
print("Total points:", df["total_points"].sum())
print("Report saved to:", output)
print("\nImages with the most annotated points:")
print(df.sort_values("total_points", ascending=False).head(10).to_string(index=False))
