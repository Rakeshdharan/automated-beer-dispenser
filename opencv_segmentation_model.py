"""
Approach 03 – Segmentation Model (YOLO + Beer/Foam Masks)
--------------------------------------------------------
This script uses a YOLO segmentation model to detect the glass,
then applies HSV-based thresholds to identify beer and foam regions
inside the segmented glass. Percentages of fill, beer, and foam
are computed and displayed on the video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys


# -------------------------------
# Configuration
# -------------------------------

# HSV thresholds for beer and foam (tuned experimentally)
LOWER_BEER = np.array([0, 70, 70])
UPPER_BEER = np.array([80, 149, 199])

LOWER_FOAM = np.array([81, 150, 200])
UPPER_FOAM = np.array([255, 255, 255])


# -------------------------------
# Utility Functions
# -------------------------------

def segment_glass(model, image):
    """
    Run YOLO segmentation model to detect glass.
    
    Args:
        model (YOLO): Pre-trained YOLO segmentation model.
        image (np.ndarray): Input frame (resized).
    
    Returns:
        list: List of binary masks for detected glass regions.
    """
    results = model(image)
    if results[0].masks:
        return results[0].masks.data.cpu().numpy()
    return []


def extract_masks(image, glass_mask):
    """
    Extract beer and foam masks from the glass region.

    Args:
        image (np.ndarray): Input frame.
        glass_mask (np.ndarray): Binary glass segmentation mask.

    Returns:
        tuple: (beer_mask, foam_mask, glass_gray)
    """
    # Apply mask to get glass-only region
    mask_resized = cv2.resize(glass_mask, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized * 255).astype(np.uint8)

    glass_only = cv2.bitwise_and(image, image, mask=mask_binary)
    glass_gray = cv2.cvtColor(glass_only, cv2.COLOR_BGR2GRAY)

    # Beer & foam segmentation
    beer_mask = cv2.inRange(glass_only, LOWER_BEER, UPPER_BEER)
    foam_mask = cv2.inRange(glass_only, LOWER_FOAM, UPPER_FOAM)

    return beer_mask, foam_mask, glass_gray, glass_only


def calculate_percentages(beer_mask, foam_mask, glass_gray):
    """
    Calculate beer, foam, and fill percentages.

    Args:
        beer_mask (np.ndarray): Mask for beer regions.
        foam_mask (np.ndarray): Mask for foam regions.
        glass_gray (np.ndarray): Gray image of segmented glass.

    Returns:
        dict: {'fill': float, 'beer': float, 'foam': float}
    """
    # Filter contours to remove noise
    b_contours, _ = cv2.findContours(beer_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h_contours, _ = cv2.findContours(foam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_beer = [cnt for cnt in b_contours if cv2.contourArea(cnt) >= 200]
    filtered_foam = [cnt for cnt in h_contours if cv2.contourArea(cnt) >= 200]

    # Fill contours to get final masks
    fill_beer = np.zeros_like(beer_mask)
    fill_foam = np.zeros_like(foam_mask)

    cv2.drawContours(fill_beer, filtered_beer, -1, color=255, thickness=cv2.FILLED)
    cv2.drawContours(fill_foam, filtered_foam, -1, color=255, thickness=cv2.FILLED)

    # Morphology for smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    filled_beer = cv2.dilate(fill_beer, kernel, iterations=1)
    filled_foam = cv2.dilate(fill_foam, kernel, iterations=1)

    # Area calculation
    glass_area = cv2.countNonZero(glass_gray)
    beer_area = cv2.countNonZero(filled_beer)
    foam_area = cv2.countNonZero(filled_foam)

    if glass_area > 0:
        fill_percentage = ((beer_area + foam_area) / glass_area) * 100
        foam_percentage = (foam_area / glass_area) * 100
        beer_percentage = (beer_area / glass_area) * 100
    else:
        fill_percentage = foam_percentage = beer_percentage = 0

    return {
        "fill": fill_percentage,
        "beer": beer_percentage,
        "foam": foam_percentage
    }


def annotate_frame(frame, percentages, prev_fill):
    """
    Annotate frame with beer/foam/fill percentages.

    Args:
        frame (np.ndarray): Image to draw on.
        percentages (dict): Percentages of fill, beer, and foam.
        prev_fill (float): Previous fill percentage (for smoothing).

    Returns:
        tuple: (annotated_frame, updated_prev_fill)
    """
    fill = percentages["fill"]
    beer = percentages["beer"]
    foam = percentages["foam"]

    # Prevent decreasing fill % (stabilization)
    if fill < prev_fill:
        fill = prev_fill
    else:
        prev_fill = fill

    # Handle case when glass is nearly full
    if fill >= 99:
        total = max(beer + foam, 1e-6)
        beer = (beer / total) * 100
        foam = (foam / total) * 100
        fill = 100

    # Annotate
    cv2.putText(frame, f"Fill: {fill:.0f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Foam: {foam:.0f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Beer: {beer:.0f}%", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, prev_fill


# -------------------------------
# Main Loop
# -------------------------------

def main(video_path, model_path="last-2.pt"):
    """
    Run beer/foam/glass segmentation pipeline on video.

    Args:
        video_path (str): Path to video file.
        model_path (str): Path to YOLO segmentation model.
    """
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    prev_fill = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.resize(frame, (320, 320))
        masks = segment_glass(model, image)

        if len(masks) > 0:
            for mask in masks:
                beer_mask, foam_mask, glass_gray, glass_only = extract_masks(image, mask)
                percentages = calculate_percentages(beer_mask, foam_mask, glass_gray)
                annotated, prev_fill = annotate_frame(glass_only, percentages, prev_fill)
                cv2.imshow("Glass Only", annotated)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opencv_segmentation_model.py <video_path>")
    else:
        main(sys.argv[1])
