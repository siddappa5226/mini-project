
import cv2
from pathlib import Path

dataset = Path(r"C:\Users\sidda\Downloads\images_and_labels\images_and_labels")

image_name = "71866.tiff"
image_path = dataset / image_name
annotation_path = dataset / "71866.txt"

image = cv2.imread(str(image_path))

if image is None:
    print("Image not found. Check the dataset path.")
    raise SystemExit

with open(annotation_path, "r") as file:
    lines = file.readlines()[1:]

count = 0

for line in lines:
    values = line.split()

    if len(values) >= 4:
        x = int(values[1])
        y = int(values[2])

        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        count += 1

print("Annotated points displayed:", count)

cv2.imshow("Corneal Cells - Annotations", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
