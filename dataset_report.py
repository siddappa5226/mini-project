
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "corneal_image_features.csv"

df = pd.read_csv(CSV_FILE)

print("CORNEAL DATASET REPORT")
print("----------------------")
print("Total images:", len(df))
print("Images with annotated points:", (df["annotated_cell_points"] > 0).sum())
print("Images with zero annotated points:", (df["annotated_cell_points"] == 0).sum())
print("Total annotated cell points:", df["annotated_cell_points"].sum())
print("Average points per image:", round(df["annotated_cell_points"].mean(), 2))

print("\nImages with the fewest annotated points:")
print(
    df[["image", "annotated_cell_points"]]
    .sort_values("annotated_cell_points")
    .head(10)
    .to_string(index=False)
)

report_path = PROJECT_DIR / "dataset_report.csv"
df[["image", "annotated_cell_points"]].to_csv(report_path, index=False)

print("\nReport saved to:", report_path)

