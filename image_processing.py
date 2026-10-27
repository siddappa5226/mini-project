import cv2
import numpy as np

def preprocess_image(image_rgb):
    """Return a displayable RGB image and a denoised grayscale image."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Empty image.")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    # CLAHE can improve local contrast; it does not establish biological damage.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    processed_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return processed_rgb, enhanced

def extract_features(gray):
    """Extract basic, generic image features for a learning prototype."""
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.size == 0:
        raise ValueError("Empty grayscale image.")
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().astype(float)
    hist = hist / max(hist.sum(), 1.0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    features = {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "edge_density": edge_density,
        "laplacian_variance": laplacian_var,
    }
    for i, value in enumerate(hist):
        features[f"hist_{i:02d}"] = float(value)
    return features
