
from pathlib import Path
import pandas as pd

# Project dataset paths
root = Path(__file__).parent
input_file = root / "dataset" / "training_features.csv"
output_file = root / "dataset" / "dataset_quality_report.csv"

# Load extracted features
df = pd.read_csv(input_file)

# Calculate dataset quality statistics
total_images = len(df)
annotated_images = int((df["cell_point_count"] > 0).sum())
images_without_points = int((df["cell_point_count"] == 0).sum())
missing_grades = int(df["grade"].isna().sum())
missing_distances = int(df["mean_nearest_point_distance"].isna().sum())

# Create a summary report
report = pd.DataFrame({
    "Metric": [
        "Total images",
        "Images with annotated points",
        "Images without annotated points",
        "Annotation coverage (%)",
        "Missing damage grades",
        "Missing nearest-point distances",
        "Average annotated points per image",
    ],
    "Value": [
        total_images,
        annotated_images,
        images_without_points,
        round(annotated_images / total_images * 100, 2)
        if total_images else 0,
        missing_grades,
        missing_distances,
        round(df["cell_point_count"].mean(), 2),
    ],
})

report.to_csv(output_file, index=False)

print("Dataset quality report created!")
print(f"Saved to: {output_file}")
print()
print(report.to_string(index=False))
