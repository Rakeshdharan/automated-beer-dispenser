"""
Approach 04 – Green-Screen Removal + Interactive Beer/Foam Calibration
----------------------------------------------------------------------
This script removes a green background from the input video, then tracks the glass
and segments beer + foam using interactive RGB thresholds. Calibration (empty/full)
yields absolute fill percentages for beer, foam and total; values are smoothed
with a rolling average and overlaid on the output video. 

"""

import cv2
import numpy as np
import json
from collections import deque
import sys

# -----------------------------
# Green Screen Remover Class
# -----------------------------
class GreenScreenRemover:
    """
    Green Screen Remover

    Removes a green background from a video and writes the output to a new file.

    Attributes:
        input_path (str): Path to input video.
        output_path (str): Path to save processed video.
        lower_green (np.array): Lower HSV threshold for green detection.
        upper_green (np.array): Upper HSV threshold for green detection.
    """

    def __init__(self, input_path, output_path="forinput.webm"):
        self.input_path = input_path
        self.output_path = output_path

        # Refined green HSV range (adjustable)
        self.lower_green = np.array([57, 155, 110])
        self.upper_green = np.array([65, 237, 200])

        # Video capture
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Cannot open video {self.input_path}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Define output writer (VP8 WebM format)
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )

    def run(self):
        """Process the video: remove green background and save output. Returns output path."""
        kernel = np.ones((3, 3), np.uint8)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Mask for green
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            # Apply black background where green detected
            bg_mask = mask == 255
            frame[bg_mask] = [0, 0, 0]

            # Write frame to output video
            self.out.write(frame)

        self.cap.release()
        self.out.release()

        # Return path to processed video
        return self.output_path

    def set_hsv_range(self, lower, upper):
        """Update HSV range for green detection."""
        self.lower_green = np.array(lower)
        self.upper_green = np.array(upper)



"""
Beer and Foam Fill-Level Detector with Calibration

- Tracks a glass in the video and detects beer + foam regions using interactive RGB thresholds.
- calibration to compute absolute fill percentage based on empty/full references.

Methodology:
1. Initialize glass tracking using edge detection and contours.
2. Align subsequent frames to the reference frame via ECC motion estimation.
3. Apply interactive RGB thresholding (trackbars) to detect beer and foam regions.
4. Clean masks using morphological operations and contour filtering.
5. Compute beer and foam top surfaces inside the glass contour.
6. If calibrated:
   - Use empty/full reference heights for absolute fill percentage.
   - Split into beer % and foam % based on detected regions.
   - Smooth values with a rolling average for stability.
7. If not calibrated:
   - Estimate fill % as (beer + foam area) / glass area.

Usage:
    python beer_fill_detector.py <video_file>

Keyboard Controls:
    Press 'e' → set empty reference height
    Press 'f' → set full reference height
    Press 'ESC' → quit
"""



class BeerFillDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.calibration_file = "calibration.json"
        self.empty_height = 0
        self.full_height = 0
        self.calibrated = False
        self.cap = None
        self.first_frame = None
        self.hull = None
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        self.first_gray = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.beer_history = deque(maxlen=5)
        self.foam_history = deque(maxlen=5)
        self.total_history = deque(maxlen=5)

        self.current_beer_pct = 0.0
        self.current_foam_pct = 0.0
        self.current_total_pct = 0.0

    def save_calibration(self):
        data = {
            "empty_height": int(self.empty_height),
            "full_height": int(self.full_height)
        }
        with open(self.calibration_file, "w") as f:
            json.dump(data, f)

    def load_calibration(self):
        try:
            with open(self.calibration_file, "r") as f:
                data = json.load(f)
                self.empty_height = data.get("empty_height", 0)
                self.full_height = data.get("full_height", 0)
                if self.empty_height and self.full_height:
                    self.calibrated = True
                    print("Calibration loaded.")
        except:
            print("Calibration not found or invalid.")

    def initialize_glass_tracker(self):
        self.cap = cv2.VideoCapture(self.video_path)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 200)
        ret, self.first_frame = self.cap.read()
        self.first_frame = cv2.resize(self.first_frame, (640, 360))
        if not ret:
            raise RuntimeError("Failed to read frame for initialization")
        gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        self.hull = box.astype(np.float32)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.first_gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)

    def create_trackbars(self):
        cv2.namedWindow('Beer Detection')
        cv2.createTrackbar('Beer Low B', 'Beer Detection', 0, 255, lambda x: None)
        cv2.createTrackbar('Beer Low G', 'Beer Detection', 70, 255, lambda x: None)
        cv2.createTrackbar('Beer Low R', 'Beer Detection', 70, 255, lambda x: None)
        cv2.createTrackbar('Beer High B', 'Beer Detection', 80, 255, lambda x: None)
        cv2.createTrackbar('Beer High G', 'Beer Detection', 149, 255, lambda x: None)
        cv2.createTrackbar('Beer High R', 'Beer Detection', 199, 255, lambda x: None)
        cv2.createTrackbar('Foam Low B', 'Beer Detection', 81, 255, lambda x: None)
        cv2.createTrackbar('Foam Low G', 'Beer Detection', 150, 255, lambda x: None)
        cv2.createTrackbar('Foam Low R', 'Beer Detection', 200, 255, lambda x: None)
        cv2.createTrackbar('Foam High B', 'Beer Detection', 255, 255, lambda x: None)
        cv2.createTrackbar('Foam High G', 'Beer Detection', 255, 255, lambda x: None)
        cv2.createTrackbar('Foam High R', 'Beer Detection', 255, 255, lambda x: None)

    def get_thresholds(self):
        beer_low = np.array([cv2.getTrackbarPos('Beer Low B', 'Beer Detection'),
                             cv2.getTrackbarPos('Beer Low G', 'Beer Detection'),
                             cv2.getTrackbarPos('Beer Low R', 'Beer Detection')])
        beer_high = np.array([cv2.getTrackbarPos('Beer High B', 'Beer Detection'),
                              cv2.getTrackbarPos('Beer High G', 'Beer Detection'),
                              cv2.getTrackbarPos('Beer High R', 'Beer Detection')])
        foam_low = np.array([cv2.getTrackbarPos('Foam Low B', 'Beer Detection'),
                             cv2.getTrackbarPos('Foam Low G', 'Beer Detection'),
                             cv2.getTrackbarPos('Foam Low R', 'Beer Detection')])
        foam_high = np.array([cv2.getTrackbarPos('Foam High B', 'Beer Detection'),
                              cv2.getTrackbarPos('Foam High G', 'Beer Detection'),
                              cv2.getTrackbarPos('Foam High R', 'Beer Detection')])
        return beer_low, beer_high, foam_low, foam_high

    def remove_small_blobs(self, mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
        return clean_mask

    def find_top_surface(self, mask, glass_mask, min_pixels_per_row=20):
        h, w = mask.shape
        combined = cv2.bitwise_and(mask, glass_mask)
        for y in range(h):
            if np.count_nonzero(combined[y, :]) >= min_pixels_per_row:
                return y
        return h

    def get_current_fill_percentages(self):
        return self.current_beer_pct, self.current_foam_pct, self.current_total_pct

    def run(self):
        self.load_calibration()
        self.initialize_glass_tracker()
        self.create_trackbars()

        width, height = 640, 360
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("beer_fill_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

            frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                _, self.warp_matrix = cv2.findTransformECC(self.first_gray, gray, self.warp_matrix, cv2.MOTION_EUCLIDEAN, self.criteria)
                transformed_hull = cv2.transform(self.hull.reshape(-1, 1, 2), self.warp_matrix).astype(np.int32)
                glass_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(glass_mask, [transformed_hull], 255)
            except:
                print("Tracking lost - using previous position")
                glass_mask = np.zeros_like(gray)

            key = cv2.waitKey(30)
            if key == ord('e'):
                self.empty_height = np.max(transformed_hull[:, 0, 1])
                print(f"Empty height set to: {self.empty_height}")
                self.save_calibration()
            elif key == ord('f'):
                self.full_height = np.min(transformed_hull[:, 0, 1])
                print(f"Full height set to: {self.full_height}")
                self.calibrated = True
                self.save_calibration()
            elif key == 27:
                break

            beer_low, beer_high, foam_low, foam_high = self.get_thresholds()
            beer_mask = cv2.inRange(frame, beer_low, beer_high)
            foam_mask = cv2.inRange(frame, foam_low, foam_high)
            beer_mask = cv2.bitwise_and(beer_mask, glass_mask)
            foam_mask = cv2.bitwise_and(foam_mask, glass_mask)
            beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_OPEN, self.kernel)
            foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, self.kernel)
            beer_mask = self.remove_small_blobs(beer_mask)
            foam_mask = self.remove_small_blobs(foam_mask)

            beer_area = cv2.countNonZero(beer_mask)
            foam_area = cv2.countNonZero(foam_mask)
            total_area = beer_area + foam_area
            glass_area = cv2.countNonZero(glass_mask)

            if self.calibrated:
                beer_top = self.find_top_surface(beer_mask, glass_mask, 10)
                foam_top = self.find_top_surface(foam_mask, glass_mask, 20)
                liquid_top = min(beer_top, foam_top)
                total_pct = 100 * (self.empty_height - liquid_top) / (self.empty_height - self.full_height)
                total_pct = np.clip(total_pct, 0, 100)
                if total_area > 0:
                    beer_pct = total_pct * (beer_area / total_area)
                    foam_pct = total_pct * (foam_area / total_area)
                else:
                    beer_pct = foam_pct = 0
            else:
                beer_pct = (beer_area / glass_area) * 100 if glass_area else 0
                foam_pct = (foam_area / glass_area) * 100 if glass_area else 0
                total_pct = beer_pct + foam_pct
                if total_pct > 100:
                    scale = 100 / total_pct
                    beer_pct *= scale
                    foam_pct *= scale
                    total_pct = 100

            self.beer_history.append(beer_pct)
            self.foam_history.append(foam_pct)
            self.total_history.append(total_pct)
            beer_pct = np.mean(self.beer_history)
            foam_pct = np.mean(self.foam_history)
            total_pct = np.mean(self.total_history)

            self.current_beer_pct = beer_pct / 100.0
            self.current_foam_pct = foam_pct / 100.0
            self.current_total_pct = total_pct / 100.0

            overlay = frame.copy()
            if foam_area:
                foam_top = self.find_top_surface(foam_mask, glass_mask, 20)
                cv2.line(overlay, (0, foam_top), (width, foam_top), (0, 255, 0), 2)
            if beer_area:
                beer_top = self.find_top_surface(beer_mask, glass_mask, 10)
                cv2.line(overlay, (0, beer_top), (width, beer_top), (0, 165, 255), 2)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Beer Fill: {beer_pct:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            cv2.putText(frame, f"Foam Fill: {foam_pct:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Fill: {total_pct:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Beer Fill Detection", frame)
            out.write(frame)

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

        return self.current_beer_pct, self.current_foam_pct, self.current_total_pct
    

if __name__ == "__main__":
    # Step 1: Read input video path from command line
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_video_path>")
        sys.exit(1)

    input_video_path = sys.argv[1]

    # Step 2: Remove green screen from the video
    green_screen_remover = GreenScreenRemover(input_video_path)
    processed_video_path = green_screen_remover.run()  # returns path to processed video

    # Step 3: Detect beer fill using the processed video
    detector = BeerFillDetector(processed_video_path)
    beer, foam, total = detector.run()

    # Step 4: Print results
    print(f"Final Beer Fill: {beer:.2f}")
    print(f"Final Foam Fill: {foam:.2f}")
    print(f"Final Total Fill: {total:.2f}")
