import cv2
import numpy as np
from ultralytics import YOLO
import sys 

def remove_small_blobs(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean, [cnt], -1, 255, cv2.FILLED)
    return clean

def overlay_mask(frame, mask, color, alpha=0.4):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, frame, 1.0, 0, frame)

def has_continuous_run(row, min_length):
    max_run = 0
    current_run = 0
    for pixel in row:
        if pixel > 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run >= min_length

def main(video_path):
    model = YOLO("withblackbest.pt")
    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("beer_debug_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Fixed HSV values
    beer_low = np.array([19, 128, 102])
    beer_high = np.array([26, 255, 248])
    foam_low = np.array([17, 14, 127])
    foam_high = np.array([29, 59, 255])

    min_foam_ratio = 0.01  # 1% of image width
    min_foam_width = int(w * min_foam_ratio)
    min_continuous_width = int(w * 0.05)  # 5% of width for continuous run in top foam line

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # YOLO segmentation
        small = cv2.resize(frame, (640, 384))
        results = model(small, verbose=False)[0]
        glass_mask = np.zeros((h, w), np.uint8)
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            glass_masks = []
            for i, cls in enumerate(classes):
                if cls == 0:
                    m = masks[i]
                    M = cv2.resize(m, (w, h))
                    glass_masks.append(M > 0.5)
            if glass_masks:
                glass_mask = (np.any(glass_masks, axis=0).astype(np.uint8) * 255)

        # Segment beer and foam
        beer_mask = cv2.inRange(hsv, beer_low, beer_high)
        foam_mask = cv2.inRange(hsv, foam_low, foam_high)
        beer_mask = cv2.bitwise_and(beer_mask, glass_mask)
        foam_mask = cv2.bitwise_and(foam_mask, glass_mask)
        beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_OPEN, kernel)
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, kernel)
        beer_mask = remove_small_blobs(beer_mask)
        foam_mask = remove_small_blobs(foam_mask)

        top_foam_line = None
        for y in range(h):
            row = foam_mask[y, :]
            if has_continuous_run(row, min_continuous_width):
                top_foam_line = y
                break

        # Robust foam bottom line 
        bottom_foam_line = None
        for y in range(h - 1, -1, -1):
            if np.count_nonzero(foam_mask[y, :]) >= min_foam_width:
                bottom_foam_line = y
                break

        # Robust beer top line 
        top_beer_line = None
        for y in range(h):
            if np.count_nonzero(beer_mask[y, :]) >= min_foam_width:
                top_beer_line = y
                break

        # Interface line (if both foam bottom and beer top exist)
        interface_line = None
        if bottom_foam_line is not None and top_beer_line is not None:
            interface_line = (bottom_foam_line + top_beer_line) // 2

        # Draw lines on the frame
        if top_foam_line is not None:
            cv2.line(frame, (0, top_foam_line), (w - 1, top_foam_line), (0, 255, 0), 2)  # Green top foam line

        if interface_line is not None:
            cv2.line(frame, (0, interface_line), (w - 1, interface_line), (0, 0, 255), 2)  # Red interface line

        # Create updated foam and beer masks within the glass region
        foam_mask_new = np.zeros_like(glass_mask)
        beer_mask_new = np.zeros_like(glass_mask)

        # Define bottom limit (start of actual liquid area)
        liquid_start_line = h - 20  
        if top_foam_line is not None:
            foam_bottom = interface_line if interface_line is not None else liquid_start_line
            foam_mask_new[top_foam_line:foam_bottom, :] = 255
            foam_mask_new = cv2.bitwise_and(foam_mask_new, glass_mask)
            foam_mask_new[liquid_start_line:, :] = 0  # cut off below thick bottom

        if interface_line is not None:
            beer_mask_new[interface_line:liquid_start_line, :] = 255
            beer_mask_new = cv2.bitwise_and(beer_mask_new, glass_mask)

        # Replace old masks
        foam_mask = foam_mask_new
        beer_mask = beer_mask_new

        # Fill calculation
        glass_area = cv2.countNonZero(glass_mask)
        beer_area = cv2.countNonZero(beer_mask)
        foam_area = cv2.countNonZero(foam_mask)
        if glass_area > 0:
            beer_pct = beer_area / glass_area * 100
            foam_pct = foam_area / glass_area * 100
            total_pct = beer_pct + foam_pct
            if total_pct > 100:
                scale = 100 / total_pct
                beer_pct *= scale
                foam_pct *= scale
        else:
            beer_pct = foam_pct = total_pct = 0

        overlay_mask(frame, glass_mask, (255, 0, 0), alpha=0.3)
        overlay_mask(frame, beer_mask, (0, 140, 255), alpha=0.4)
        overlay_mask(frame, foam_mask, (0, 255, 0), alpha=0.4)

        # Text
        cv2.putText(frame, f"Beer: {beer_pct:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        cv2.putText(frame, f"Foam: {foam_pct:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Fill: {total_pct:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        # Show
        cv2.imshow("Combined View", frame)
        cv2.imshow("Glass Mask", glass_mask)
        cv2.imshow("Beer Mask", beer_mask)
        cv2.imshow("Foam Mask", foam_mask)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    main(video_path)
