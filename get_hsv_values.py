import cv2
import numpy as np

def nothing(x):
    pass

# Load sample image 
img = cv2.imread('first_frame.png')  #
img = cv2.resize(img, (640, 360))

cv2.namedWindow('Beer HSV Mask')
cv2.namedWindow('Foam HSV Mask')
cv2.namedWindow('Glass HSV Mask')

# Beer HSV Trackbars
cv2.createTrackbar('Beer_H_low', 'Beer HSV Mask', 10, 180, nothing)
cv2.createTrackbar('Beer_S_low', 'Beer HSV Mask', 80, 255, nothing)
cv2.createTrackbar('Beer_V_low', 'Beer HSV Mask', 60, 255, nothing)
cv2.createTrackbar('Beer_H_high','Beer HSV Mask', 30, 180, nothing)
cv2.createTrackbar('Beer_S_high','Beer HSV Mask', 255, 255, nothing)
cv2.createTrackbar('Beer_V_high','Beer HSV Mask', 220, 255, nothing)

# Foam HSV Trackbars
cv2.createTrackbar('Foam_H_low', 'Foam HSV Mask', 0, 180, nothing)
cv2.createTrackbar('Foam_S_low', 'Foam HSV Mask', 0, 255, nothing)
cv2.createTrackbar('Foam_V_low', 'Foam HSV Mask', 160, 255, nothing)
cv2.createTrackbar('Foam_H_high','Foam HSV Mask', 180, 180, nothing)
cv2.createTrackbar('Foam_S_high','Foam HSV Mask', 80, 255, nothing)
cv2.createTrackbar('Foam_V_high','Foam HSV Mask', 255, 255, nothing)

# Glass HSV Trackbars (useful if glass has color, e.g. brown glass)
cv2.createTrackbar('Glass_H_low', 'Glass HSV Mask', 0, 180, nothing)
cv2.createTrackbar('Glass_S_low', 'Glass HSV Mask', 0, 255, nothing)
cv2.createTrackbar('Glass_V_low', 'Glass HSV Mask', 70, 255, nothing)
cv2.createTrackbar('Glass_H_high','Glass HSV Mask', 180, 180, nothing)
cv2.createTrackbar('Glass_S_high','Glass HSV Mask', 80, 255, nothing)
cv2.createTrackbar('Glass_V_high','Glass HSV Mask', 255, 255, nothing)

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get trackbar values for Beer
    beer_hl = cv2.getTrackbarPos('Beer_H_low', 'Beer HSV Mask')
    beer_sl = cv2.getTrackbarPos('Beer_S_low', 'Beer HSV Mask')
    beer_vl = cv2.getTrackbarPos('Beer_V_low', 'Beer HSV Mask')
    beer_hh = cv2.getTrackbarPos('Beer_H_high','Beer HSV Mask')
    beer_sh = cv2.getTrackbarPos('Beer_S_high','Beer HSV Mask')
    beer_vh = cv2.getTrackbarPos('Beer_V_high','Beer HSV Mask')
    beer_low = np.array([beer_hl, beer_sl, beer_vl])
    beer_high= np.array([beer_hh, beer_sh, beer_vh])

    # Get trackbar values for Foam
    foam_hl = cv2.getTrackbarPos('Foam_H_low', 'Foam HSV Mask')
    foam_sl = cv2.getTrackbarPos('Foam_S_low', 'Foam HSV Mask')
    foam_vl = cv2.getTrackbarPos('Foam_V_low', 'Foam HSV Mask')
    foam_hh = cv2.getTrackbarPos('Foam_H_high','Foam HSV Mask')
    foam_sh = cv2.getTrackbarPos('Foam_S_high','Foam HSV Mask')
    foam_vh = cv2.getTrackbarPos('Foam_V_high','Foam HSV Mask')
    foam_low = np.array([foam_hl, foam_sl, foam_vl])
    foam_high= np.array([foam_hh, foam_sh, foam_vh])

    # Get trackbar values for Glass
    glass_hl = cv2.getTrackbarPos('Glass_H_low', 'Glass HSV Mask')
    glass_sl = cv2.getTrackbarPos('Glass_S_low', 'Glass HSV Mask')
    glass_vl = cv2.getTrackbarPos('Glass_V_low', 'Glass HSV Mask')
    glass_hh = cv2.getTrackbarPos('Glass_H_high','Glass HSV Mask')
    glass_sh = cv2.getTrackbarPos('Glass_S_high','Glass HSV Mask')
    glass_vh = cv2.getTrackbarPos('Glass_V_high','Glass HSV Mask')
    glass_low = np.array([glass_hl, glass_sl, glass_vl])
    glass_high= np.array([glass_hh, glass_sh, glass_vh])

    # Create each mask
    mask_beer = cv2.inRange(hsv, beer_low, beer_high)
    mask_foam = cv2.inRange(hsv, foam_low, foam_high)
    mask_glass= cv2.inRange(hsv, glass_low, glass_high)

    cv2.imshow('Beer HSV Mask', mask_beer)
    cv2.imshow('Foam HSV Mask', mask_foam)
    cv2.imshow('Glass HSV Mask', mask_glass)

    key = cv2.waitKey(30)
    if key == ord('s'): 
        print("Beer HSV low:", beer_low, "high:", beer_high)
        print("Foam HSV low:", foam_low, "high:", foam_high)
        print("Glass HSV low:", glass_low, "high:", glass_high)
    if key == 27: 
        break
cv2.destroyAllWindows()