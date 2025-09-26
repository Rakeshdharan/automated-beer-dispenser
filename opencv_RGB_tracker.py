import cv2
import numpy as np
import time
import sys

def nothing(x):
    """Dummy callback function for OpenCV trackbars."""
    pass

def detect_beer_with_sliders(video_path):
    """
    Detect beer and foam regions in a glass using adjustable RGB thresholds via OpenCV trackbars.

    Args:
        video_path (str): Path to the input video file.

    Workflow:
        1. Load video stream.
        2. Create interactive trackbars for adjusting beer and foam color thresholds.
        3. Segment beer and foam regions in each frame using color masking.
        4. Identify glass region and calculate beer fill percentage.
        5. Overlay results (beer, foam, glass boundary, fill %) onto video.
    """

    # Load video
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not load video")
        return None

    # Create window for trackbars
    cv2.namedWindow('Beer Detection')

    # Beer thresholds (RGB)
    cv2.createTrackbar('Low R', 'Beer Detection', 0, 255, nothing)
    cv2.createTrackbar('Low G', 'Beer Detection', 0, 255, nothing)
    cv2.createTrackbar('Low B', 'Beer Detection', 0, 255, nothing)

    cv2.createTrackbar('High R', 'Beer Detection', 255, 255, nothing)
    cv2.createTrackbar('High G', 'Beer Detection', 255, 255, nothing)
    cv2.createTrackbar('High B', 'Beer Detection', 255, 255, nothing)

    # Foam thresholds (RGB)
    cv2.createTrackbar('Head Low R', 'Beer Detection', 200, 255, nothing)
    cv2.createTrackbar('Head Low G', 'Beer Detection', 200, 255, nothing)
    cv2.createTrackbar('Head Low B', 'Beer Detection', 200, 255, nothing)

    cv2.createTrackbar('Head High R', 'Beer Detection', 255, 255, nothing)
    cv2.createTrackbar('Head High G', 'Beer Detection', 255, 255, nothing)
    cv2.createTrackbar('Head High B', 'Beer Detection', 255, 255, nothing)

    while True:
        # Fetch video frame
        ret, image = vid.read()
        if not ret:
            break

        # Get trackbar values for thresholds
        low_r = cv2.getTrackbarPos('Low R', 'Beer Detection')
        low_g = cv2.getTrackbarPos('Low G', 'Beer Detection')
        low_b = cv2.getTrackbarPos('Low B', 'Beer Detection')

        high_r = cv2.getTrackbarPos('High R', 'Beer Detection')
        high_g = cv2.getTrackbarPos('High G', 'Beer Detection')
        high_b = cv2.getTrackbarPos('High B', 'Beer Detection')

        head_low_r = cv2.getTrackbarPos('Head Low R', 'Beer Detection')
        head_low_g = cv2.getTrackbarPos('Head Low G', 'Beer Detection')
        head_low_b = cv2.getTrackbarPos('Head Low B', 'Beer Detection')

        head_high_r = cv2.getTrackbarPos('Head High R', 'Beer Detection')
        head_high_g = cv2.getTrackbarPos('Head High G', 'Beer Detection')
        head_high_b = cv2.getTrackbarPos('Head High B', 'Beer Detection')

        # Define color ranges for beer, foam, and glass
        lower_beer = np.array([50, 100, 150])  # fixed lower threshold for beer
        upper_beer = np.array([high_b, high_g, high_r])

        lower_head = np.array([head_low_b, head_low_g, head_low_r])
        upper_head = np.array([head_high_b, head_high_g, head_high_r])

        lower_glass = np.array([0, 0, 20])
        upper_glass = np.array([255, 255, 255])

        # Create masks
        beer_mask = cv2.inRange(image, lower_beer, upper_beer)
        head_mask = cv2.inRange(image, lower_head, upper_head)
        glass_mask = cv2.inRange(image, lower_glass, upper_glass)

        # Combine masks for contour detection
        combined_mask = cv2.bitwise_or(beer_mask, head_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = image.copy()

        if contours:
            # Largest contour = glass region
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(glass_mask, [largest_contour], -1, 255, -1)

            # Refine beer & foam masks inside glass
            beer_mask = cv2.bitwise_and(beer_mask, glass_mask)
            head_mask = cv2.bitwise_and(head_mask, glass_mask)

            # Calculate fill percentage
            total_glass_area = cv2.countNonZero(glass_mask)
            beer_area = cv2.countNonZero(beer_mask)

            if total_glass_area > 0:
                fill_percentage = (beer_area / total_glass_area) * 100
            else:
                fill_percentage = 0

            # Overlay beer & foam colors
            output[beer_mask > 0] = [0, 255, 255]  # Yellow for beer
            output[head_mask > 0] = [255, 255, 0]  # Cyan for foam

            # Draw glass edges
            edges = cv2.Canny(glass_mask, 50, 150)
            output[edges > 0] = [0, 0, 255]

            # Display percentage
            cv2.putText(output, f"Fill: {fill_percentage:.1f}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show results
        cv2.imshow('Beer Detection', output)
        time.sleep(0.1)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid.release()
    cv2.destroyAllWindows()
    return output


if __name__ == "__main__":
    video_path = sys.argv[1]
    result = detect_beer_with_sliders(video_path)
