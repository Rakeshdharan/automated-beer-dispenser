"""
Approach 05 – Optical Flow + Beer/Foam HSV Segmentation
--------------------------------------------------------
This script tracks the glass contour using optical flow from the first frame,
then segments beer and foam inside the glass using HSV thresholds. 
Small noisy blobs are removed, and the fill percentage is computed 
frame-by-frame and visualized on the output video.
"""

import cv2
import numpy as np
import sys

# -------------------------------
# Utility Functions
# -------------------------------

def remove_small_blobs(mask, min_area=500):
    """
    Remove small connected components from a binary mask.

    Args:
        mask (np.ndarray): Binary mask to clean.
        min_area (int): Minimum contour area to keep.

    Returns:
        np.ndarray: Cleaned binary mask with small blobs removed.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
    return clean_mask

# -------------------------------
# Main Processing Function
# -------------------------------

def main():
    # --- Step 0: Read input video path from command line ---
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    # Video output configuration
    width, height = 640, 360
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("beer2_fill_outputk.mp4", fourcc, fps, (width, height))

    # --- Step 1: Read first frame and detect initial glass contour ---
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (width, height))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find initial glass contour
    blurred = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    glass_contour = contours[0]  # largest contour assumed to be glass
    glass_mask = np.zeros_like(prev_gray)
    cv2.drawContours(glass_mask, [glass_contour], -1, 255, cv2.FILLED)

    # Use contour points for optical flow tracking
    prev_points = glass_contour.reshape(-1, 1, 2).astype(np.float32)

    # Precompute vertical bounds of glass
    glass_top_y = np.min(glass_contour[:, :, 1])
    glass_bottom_y = np.max(glass_contour[:, :, 1])

    # Morphological kernel for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- Step 2: Frame-by-frame processing ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track glass contour using optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        # Update glass mask based on tracked points
        if len(good_new) >= 5:
            new_contour = good_new.reshape(-1, 1, 2).astype(np.int32)
            glass_mask = np.zeros_like(gray)
            cv2.drawContours(glass_mask, [new_contour], -1, 255, cv2.FILLED)
            glass_contour = new_contour
            prev_points = good_new.reshape(-1, 1, 2)
        else:
            # Fallback to previous contour if tracking fails
            cv2.drawContours(glass_mask, [glass_contour], -1, 255, cv2.FILLED)

        # --- Step 3: Beer and Foam Segmentation using HSV ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Beer mask (orange-yellow range)
        lower_beer = np.array([15, 50, 100])
        upper_beer = np.array([35, 255, 255])
        beer_mask = cv2.inRange(hsv, lower_beer, upper_beer)
        beer_mask = cv2.bitwise_and(beer_mask, glass_mask)
        beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_OPEN, kernel)
        beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_CLOSE, kernel)

        # Foam mask (white range)
        lower_foam = np.array([0, 0, 180])
        upper_foam = np.array([50, 70, 255])
        foam_mask = cv2.inRange(hsv, lower_foam, upper_foam)
        foam_mask = cv2.bitwise_and(foam_mask, glass_mask)
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, kernel)

        # Remove small noisy blobs
        beer_mask = remove_small_blobs(beer_mask, min_area=500)
        foam_mask = remove_small_blobs(foam_mask, min_area=300)

        # --- Step 4: Compute fill percentage ---
        beer_coords = np.column_stack(np.where(beer_mask > 0))
        foam_coords = np.column_stack(np.where(foam_mask > 0))

        if len(beer_coords) == 0:
            fill_percent = 0.0
            beer_level = glass_bottom_y
        else:
            beer_level = np.min(beer_coords[:, 0])
            fill_percent = (glass_bottom_y - beer_level) / (glass_bottom_y - glass_top_y) * 100
            fill_percent = np.clip(fill_percent, 0, 100)

        foam_level = np.min(foam_coords[:, 0]) if len(foam_coords) > 0 else beer_level

        # --- Step 5: Visualization ---
        overlay = frame.copy()
        cv2.drawContours(overlay, [glass_contour], -1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Fill: {fill_percent:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(overlay)
        cv2.imshow("Beer Fill Tracking", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame for optical flow
        prev_gray = gray.copy()

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Output saved to: beer2_fill_outputk.mp4")

if __name__ == "__main__":
    main()
