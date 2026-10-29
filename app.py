
from pathlib import Path
import math
import statistics

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


from image_processing import preprocess_image, extract_features


# ==================================================
# 1. PROJECT PATHS AND PAGE CONFIGURATION
# ==================================================

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_DIR = IMAGE_DIR / "images_and_labels"
FEATURES_PATH = DATASET_DIR / "training_features.csv"
MODEL_PATH = ROOT / "models" / "damage_model.joblib"


st.set_page_config(
    page_title="Corneal Preservation AI",
    page_icon="🔬",
    layout="wide"
)

st.title("Corneal Preservation AI Dashboard")

st.caption(
    "AI-Based Prediction of Grading of Hypothermia-Induced "
    "Donor Corneal Endothelial Damage and Recovery"
)

st.warning(
    "Research prototype only. This dashboard is not clinically "
    "validated and must not be used for diagnosis or transplant decisions."
)


# ==================================================
# 2. ANNOTATION FUNCTIONS
# ==================================================

def load_annotations(annotation_path, width, height):
    """Load valid x, y, tag points from a matching text file."""

    points = []

    if not annotation_path.exists():
        return points

    with annotation_path.open(
        "r", encoding="utf-8-sig"
    ) as file:
        next(file, None)

        for line in file:
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                x = int(parts[1])
                y = int(parts[2])
                tag = parts[3]
            except ValueError:
                continue

            if 0 <= x < width and 0 <= y < height:
                points.append((x, y, tag))

    return points


def draw_annotations(image, points, marker_size=4):
    """Draw annotation points without implying damage severity."""

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for x, y, tag in points:
        r = marker_size

        if tag == "0":
            colour = "red"
        elif tag == "1":
            colour = "cyan"
        else:
            colour = "yellow"

        draw.line(
            (x - r, y, x + r, y),
            fill=colour,
            width=1
        )

        draw.line(
            (x, y - r, x, y + r),
            fill=colour,
            width=1
        )

    return annotated


def show_tag_counts(points):
    """Display counts for annotation tags."""

    if not points:
        st.info("No valid annotated points found.")
        return

    counts = {}

    for _, _, tag in points:
        counts[tag] = counts.get(tag, 0) + 1

    st.markdown("#### Annotation tag counts")

    tag_items = sorted(counts.items())
    columns = st.columns(max(len(tag_items), 1))

    for column, (tag, count) in zip(columns, tag_items):
        with column:
            st.metric(f"Tag {tag}", count)

    st.caption(
        "Annotation tags are categories in the supplied dataset. "
        "They are not damage grades unless confirmed by the dataset documentation."
    )


def calculate_nearest_point_distance(points):
    """Calculate mean nearest-point distance, if possible."""

    if len(points) < 2:
        return None

    nearest_distances = []

    for i, (x, y, _) in enumerate(points):
        nearest = min(
            math.hypot(x - other_x, y - other_y)
            for j, (other_x, other_y, _) in enumerate(points)
            if i != j
        )

        nearest_distances.append(nearest)

    return statistics.mean(nearest_distances)


# ==================================================
# 3. IMAGE UPLOAD AND IMAGE ANALYSIS
# ==================================================

st.divider()
st.header("1. Microscopy Image Analysis")

uploaded = st.file_uploader(
    "Upload a corneal endothelial microscopy image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key="uploaded_cornea_image"
)

if uploaded is not None:
    try:
        pil_image = Image.open(uploaded).convert("RGB")
        image = np.array(pil_image)
        width, height = pil_image.size

        annotation_path = (
            ANNOTATION_DIR / f"{Path(uploaded.name).stem}.txt"
        )

        points = load_annotations(
            annotation_path, width, height
        )

        original_col, annotated_col = st.columns(2)

        with original_col:
            st.image(
                pil_image,
                caption="Original uploaded image",
                use_container_width=True
            )

        with annotated_col:
            if annotation_path.exists():
                annotated_image = draw_annotations(
                    pil_image, points
                )

                st.image(
                    annotated_image,
                    caption="Image with annotation points",
                    use_container_width=True
                )
            else:
                st.image(
                    pil_image,
                    caption="No matching annotation file found",
                    use_container_width=True
                )

        if annotation_path.exists():
            st.caption(f"Annotation file: {annotation_path.name}")
            st.metric("Annotated points", len(points))
            show_tag_counts(points)
        else:
            st.info(
                "No matching annotation file was found. "
                "Image analysis can continue without annotations."
            )

        # Preprocess image
        processed, gray = preprocess_image(image)

        st.subheader("2. Image Preprocessing")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original image",
                use_container_width=True
            )

        with col2:
            st.image(
                processed,
                caption="Preprocessed image",
                use_container_width=True
            )

        # Extract image features
        features = extract_features(gray)

        st.subheader("3. Extracted Image Features")

        feature_df = pd.DataFrame(
            list(features.items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            feature_df,
            hide_index=True,
            use_container_width=True
        )

        # Optional model prediction
        st.subheader("4. AI Damage-Grade Estimate")

        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)

                if hasattr(model, "feature_names_in_"):
                    model_columns = list(model.feature_names_in_)
                    row = pd.DataFrame([features]).reindex(
                        columns=model_columns
                    )

                    missing_features = [
                        name for name in model_columns
                        if name not in features
                    ]
                else:
                    row = pd.DataFrame([features])
                    missing_features = []

                if missing_features:
                    st.error(
                        "The model requires features that the image "
                        f"processor does not provide: {missing_features}"
                    )

                elif row.isna().any().any():
                    st.error(
                        "Some required model feature values are missing. "
                        "Check the training and preprocessing pipelines."
                    )

                else:
                    prediction = model.predict(row)[0]

                    st.metric(
                        "Saved model output",
                        str(prediction)
                    )

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(row)[0]

                        probability_df = pd.DataFrame({
                            "Class": model.classes_,
                            "Model score": probabilities
                        })

                        st.dataframe(
                            probability_df,
                            hide_index=True,
                            use_container_width=True
                        )

                    st.caption(
                        "This output is experimental. Model scores are "
                        "not validated clinical probabilities. Confirm "
                        "training labels and independent evaluation results."
                    )

            except Exception as exc:
                st.error(f"Could not run the saved model: {exc}")

        else:
            st.info(
                "No saved damage-grade model was found. Verified expert "
                "damage-grade labels are required before training."
            )

    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")

else:
    st.info("Upload an image to begin.")


# ==================================================
# 4. STORAGE TEMPERATURE MONITORING
# ==================================================

st.divider()
st.header("5. Storage Temperature")

mode = st.radio(
    "Temperature source",
    ["Demo / simulated", "Serial (Arduino)"],
    horizontal=True
)

temp = None
threshold_low = 2.0
threshold_high = 8.0

if mode == "Demo / simulated":
    temp = st.number_input(
        "Demo temperature (°C)",
        value=4.1,
        step=0.1,
        format="%.2f"
    )

    threshold_low = st.number_input(
        "Lower configured limit (°C)",
        value=2.0,
        step=0.5
    )

    threshold_high = st.number_input(
        "Upper configured limit (°C)",
        value=8.0,
        step=0.5
    )

    st.caption(
        "Demo readings and limits are placeholders. Use a calibrated "
        "sensor and limits approved for your laboratory protocol."
    )

else:
    try:
        import serial
        import serial.tools.list_ports

        ports = [
            port.device
            for port in serial.tools.list_ports.comports()
        ]

        threshold_low = st.number_input(
            "Lower configured limit (°C)",
            value=2.0,
            step=0.5
        )

        threshold_high = st.number_input(
            "Upper configured limit (°C)",
            value=8.0,
            step=0.5
        )

        if ports:
            selected_port = st.selectbox(
                "Arduino COM port",
                ports
            )

            baud = st.selectbox(
                "Baud rate",
                [9600, 115200]
            )

            if st.button("Read latest Arduino temperature"):
                try:
                    with serial.Serial(
                        selected_port,
                        baud,
                        timeout=3
                    ) as connection:
                        connection.reset_input_buffer()

                        line = connection.readline().decode(
                            "utf-8",
                            errors="ignore"
                        ).strip()

                    temp = float(line)
                    st.session_state["last_temp"] = temp

                except Exception as exc:
                    st.error(
                        f"Could not read Arduino temperature: {exc}"
                    )

            if temp is None:
                temp = st.session_state.get("last_temp")

        else:
            st.warning(
                "No serial port detected. Check the Arduino connection "
                "and close any other program using its COM port."
            )

            temp = st.session_state.get("last_temp")

    except ImportError:
        st.error(
            "PySerial is not installed. Run: "
            "python -m pip install pyserial"
        )

if temp is not None:
    st.metric("Latest temperature", f"{temp:.2f} °C")

    if threshold_low > threshold_high:
        st.error("The lower limit must not exceed the upper limit.")
    elif temp < threshold_low or temp > threshold_high:
        st.error("Temperature is outside the configured limits.")
    else:
        st.success("Temperature is within the configured limits.")

    st.caption(
        "This status compares the reading with configured limits only. "
        "It does not establish cellular health or transplant suitability."
    )
else:
    st.info("Use demo mode or connect an Arduino to view a temperature reading.")


# ==================================================
# 5. INTERPRETATION AND LIMITATIONS
# ==================================================

st.divider()
st.header("6. Interpretation")

st.write(
    "This dashboard displays microscopy images, image-derived features, "
    "annotation points, optional saved-model output, and storage temperature."
)

st.warning(
    "Temperature alone cannot establish cellular damage. Image-derived "
    "features and annotation counts are not validated damage grades."
)

st.caption(
    "Recovery prediction requires separate, verified recovery outcome labels "
    "and appropriate model evaluation. This prototype must not guide clinical decisions."
)


# ==================================================
# 6. DATASET IMAGE AND CELL-POINT ANALYSIS
# ==================================================

st.divider()
st.header("7. Dataset Image & Cell-Point Analysis")

if not IMAGE_DIR.exists():
    st.error(f"Dataset image folder not found: {IMAGE_DIR}")

else:
    image_files = sorted(
        list(IMAGE_DIR.glob("*.tiff"))
        + list(IMAGE_DIR.glob("*.tif"))
    )

    if not image_files:
        st.warning("No TIFF images found in dataset/images.")

    else:
        selected_name = st.selectbox(
            "Select a dataset image",
            [path.name for path in image_files],
            key="dataset_image_selector"
        )

        image_path = IMAGE_DIR / selected_name
        annotation_path = (
            ANNOTATION_DIR / f"{image_path.stem}.txt"
        )

        try:
            with Image.open(image_path) as source:
                original = source.convert("RGB")

            width, height = original.size

            points = load_annotations(
                annotation_path, width, height
            )

            annotated = draw_annotations(original, points)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original,
                    caption="Original dataset image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    annotated,
                    caption="Dataset image with annotation points",
                    use_container_width=True
                )

            if annotation_path.exists():
                st.caption(
                    f"Annotation file: {annotation_path.name}"
                )
                show_tag_counts(points)
            else:
                st.warning(
                    f"No matching annotation file: {annotation_path.name}"
                )

            gray = np.array(original.convert("L"))

            mean_intensity = float(np.mean(gray))
            contrast = float(np.std(gray))

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Width", f"{width} px")
            col2.metric("Height", f"{height} px")
            col3.metric("Annotated points", len(points))
            col4.metric("Mean intensity", f"{mean_intensity:.2f}")

            st.metric(
                "Contrast (standard deviation)",
                f"{contrast:.2f}"
            )

            distance = calculate_nearest_point_distance(points)

            if distance is not None:
                st.metric(
                    "Average nearest-point distance",
                    f"{distance:.2f} pixels"
                )
            elif len(points) == 1:
                st.info(
                    "Only one point is available; nearest-point distance "
                    "cannot be calculated."
                )
            else:
                st.info(
                    "No valid annotated points are available to calculate "
                    "nearest-point distance."
                )

            st.caption(
                "Research analysis only. Annotation tags are not validated "
                "damage grades or recovery predictions."
            )

        except Exception as exc:
            st.error(f"Could not analyze the dataset image: {exc}")


# ==================================================
# 7. LIVE DATASET QUALITY REPORT
# ==================================================

st.divider()
st.header("8. Dataset Quality Report")

if FEATURES_PATH.exists():
    try:
        # Read the source CSV directly on every dashboard rerun.
        # Do not read the old generated quality-report CSV here.
        quality_df = pd.read_csv(FEATURES_PATH)

        required_columns = [
            "filename",
            "grade",
            "cell_point_count",
            "mean_nearest_point_distance"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in quality_df.columns
        ]

        if missing_columns:
            st.error(
                f"Required columns are missing from training_features.csv: "
                f"{missing_columns}"
            )

        elif quality_df.empty:
            st.warning("The training features file contains no records.")

        else:
            total_images = len(quality_df)

            point_counts = pd.to_numeric(
                quality_df["cell_point_count"],
                errors="coerce"
            )

            annotated_images = int((point_counts > 0).sum())
            no_annotation_images = int((point_counts == 0).sum())

            missing_point_counts = int(point_counts.isna().sum())
            missing_grades = int(quality_df["grade"].isna().sum())
            missing_distances = int(
                quality_df["mean_nearest_point_distance"].isna().sum()
            )

            coverage = (
                annotated_images / total_images * 100
                if total_images else 0
            )

            average_points = point_counts.mean()

            # Summary metric cards
            col1, col2, col3 = st.columns(3)

            col1.metric("Total images", total_images)

            col2.metric(
                "Images with annotated points",
                annotated_images,
                f"{coverage:.2f}% coverage"
            )

            col3.metric(
                "Images without annotated points",
                no_annotation_images
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Missing damage grades",
                missing_grades
            )

            col5.metric(
                "Missing point distances",
                missing_distances
            )

            col6.metric(
                "Missing point counts",
                missing_point_counts
            )

            # Annotation coverage chart built from current data
            st.subheader("Annotation Coverage")

            chart_df = pd.DataFrame(
                {
                    "Number of images": [
                        annotated_images,
                        no_annotation_images
                    ]
                },
                index=[
                    "With annotated points",
                    "Without annotated points"
                ]
            )

            st.bar_chart(chart_df)

            # Detailed report
            st.subheader("Detailed Quality Summary")

            summary_df = pd.DataFrame({
                "Metric": [
                    "Total images",
                    "Images with annotated points",
                    "Images without annotated points",
                    "Annotation coverage (%)",
                    "Missing damage grades",
                    "Missing nearest-point distances",
                    "Missing point counts",
                    "Average annotated points per image"
                ],
                "Value": [
                    total_images,
                    annotated_images,
                    no_annotation_images,
                    round(coverage, 2),
                    missing_grades,
                    missing_distances,
                    missing_point_counts,
                    round(float(average_points), 2)
                    if pd.notna(average_points) else "Unavailable"
                ]
            })

            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Calculated directly from dataset/training_features.csv "
                "whenever this dashboard reruns. Annotation coverage does "
                "not measure endothelial damage severity."
            )

            st.info(
                "Damage-grade and recovery models require verified labels. "
                "Missing grades must not be guessed or automatically generated."
            )

    except Exception as exc:
        st.error(
            f"Could not generate the dataset quality report: {exc}"
        )

else:
    st.warning(
        "training_features.csv was not found. Expected location: "
        f"{FEATURES_PATH}"
    )


# ==================================================
# END OF DASHBOARD
# ==================================================
