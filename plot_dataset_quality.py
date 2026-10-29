
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).parent
csv_path = root / "dataset" / "training_features.csv"
output_path = root / "dataset" / "annotation_coverage.png"

df = pd.read_csv(csv_path)

annotated = int((df["cell_point_count"] > 0).sum())
not_annotated = int((df["cell_point_count"] == 0).sum())

labels = ["With annotated points", "Without annotated points"]
values = [annotated, not_annotated]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values)

plt.title("Corneal Dataset: Annotation Coverage")
plt.ylabel("Number of Images")
plt.bar_label(bars)
plt.tight_layout()

plt.savefig(output_path, dpi=200)
plt.show()

print(f"Chart saved to: {output_path}")
