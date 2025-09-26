import cv2
import numpy as np
from collections import deque

class SimpleBeerFillDetector:
    def __init__(self):
        self.lower_beer = np.array([10, 60, 80])
        self.upper_beer = np.array([50, 255, 255])
        self.lower_head = np.array([0, 0, 160])
        self.upper_head = np.array([150, 60, 255])
        self.total_glass_area = 10000
        self.fill_percentage_prev = 0

        self._last_values = deque(maxlen=5)
        self._fir_coefficients = []
        for i in range(self._last_values.maxlen):
            self._fir_coefficients += [1.0/self._last_values.maxlen]

    def update(self, frame):
        image = cv2.resize(frame, (320, 240))

        crop_image = image[10:239, 140:300] 
        hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
        
        beer_mask = cv2.inRange(hsv, self.lower_beer,  self.upper_beer)
        head_mask = cv2.inRange(hsv,  self.lower_head,  self.upper_head)
        combined_mask = cv2.bitwise_or(beer_mask, head_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            beer_area = cv2.countNonZero(beer_mask)
            head_area = cv2.countNonZero(head_mask)
            
            if head_area > 0 or beer_area > 0 :
                fill_percentage = ((beer_area + head_area) / self.total_glass_area)
                foam_percentage = (head_area / (beer_area + head_area))
            else:
                fill_percentage = 0

            self._last_values.append((fill_percentage, foam_percentage))


        fill_filtered = 0
        foam_filtered = 0
        i = 0

        for (fill, foam) in self._last_values:
            fill_filtered += self._fir_coefficients[i]*fill
            foam_filtered += self._fir_coefficients[i]*foam

        fill_filtered = min(fill_filtered, 1)
        foam_filtered = min(foam_filtered, 1) 

        #print(foam_filtered)

        cv2.putText(crop_image, f"P {str(beer_area+head_area)}/{str(self.total_glass_area)}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(crop_image, f"F {str(round(fill_filtered, 2))}/{str(round(foam_filtered, 2))}", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return fill_filtered, foam_filtered, crop_image