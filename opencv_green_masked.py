import cv2
import numpy as np
import sys

def extract_glass_with_sliders(video_path):
    """
    Extracts the glass region from a video stream and estimates the beer fill level.

    Args:
        video_path (str): Path to the input video file.

    Workflow:
        1. Reads video frames one by one.
        2. Detects green background and removes it.
        3. Identifies the glass contour using morphological operations and contour analysis.
        4. Segments beer and foam regions using HSV color ranges.
        5. Calculates fill percentage (beer + foam area relative to glass area).
        6. Displays live visualization with detected contours and fill percentage.
    """

    # Open video capture
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not load video.")
        return None

    fill_percentage_prev = 0  # Keeps track of max fill level detected

    while True:
        ret, image = vid.read()
        if not ret:
            break

        # Resize frame for consistent processing
        image = cv2.resize(image, (640, 480))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Edge detection thresholds
        canny_min = 50
        canny_max = 220

        # HSV thresholds for segmentation
        lower_green = np.array([50, 110, 50])
        upper_green = np.array([179, 255, 255])

        lower_beer = np.array([0, 70, 70])
        upper_beer = np.array([80, 149, 199])
        
        lower_head = np.array([81, 150, 200])
        upper_head = np.array([255, 255, 255])

        # Background (green) mask and cleanup
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(non_green_mask, cv2.MORPH_CLOSE, kernel)

        # Find glass contour (largest object after background removal)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glass_mask = np.zeros_like(clean_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(glass_mask, [largest], -1, 255, -1)

        # Extract glass region only
        glass_only = cv2.bitwise_and(image, image, mask=glass_mask)
        glass_gray = cv2.cvtColor(glass_only, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(glass_gray, canny_min, canny_max)

        # Segment beer and foam
        beer_mask = cv2.inRange(glass_only, lower_beer, upper_beer)
        head_mask = cv2.inRange(glass_only, lower_head, upper_head)

        # Compute areas
        glass_area = cv2.countNonZero(glass_gray)
        beer_area = cv2.countNonZero(beer_mask)
        head_area = cv2.countNonZero(head_mask)

        # Calculate fill percentage
        if glass_area > 0:
            fill_percentage = ((beer_area + head_area) * 1.1 / glass_area) * 100
        else:
            fill_percentage = 0

        # Keep track of highest detected fill percentage
        if fill_percentage > fill_percentage_prev:
            fill_percentage_prev = fill_percentage

        # Display fill percentage on frame
        cv2.putText(glass_only, f"Fill: {fill_percentage_prev:.0f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show output window
        cv2.imshow("Glass Fill Level", glass_only)

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Cleanup
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
    else:
        video_path = sys.argv[1]
        extract_glass_with_sliders(video_path)
