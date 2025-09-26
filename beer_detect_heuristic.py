import cv2
import numpy as np
from . import BeerFIR


class HeuristicBeerFillDetector:
    def __init__(self, morph_open_iterations: int = 1, morph_close_iterations: int = 5, morph_kernel_diameter: int = 5,
                 frame_size: tuple[int, int] = (256, 256), roi_size: tuple[int, int] = (150, 256),
                 otsu_scale_beer: float = 0.3, otsu_scale_foam: float = 0.3, glass_area: int = 8000):

        self._morph_open_iterations = morph_open_iterations
        self._morph_close_iterations = morph_close_iterations
        self._frame_size = frame_size
        self._roi_size = roi_size
        self._roi_center = (172, 178)
        self._otsu_scale_foam = otsu_scale_foam
        self._otsu_scale_beer = otsu_scale_beer
        self._glass_area = glass_area
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_diameter, morph_kernel_diameter)
        )

        self._fill_fir = BeerFIR()
        self._foam_fir = BeerFIR()

    def square(self, image):
        img_float = image.astype(np.float32) / 255.0
        img_squared = img_float ** 2  # element-wise square
        return (img_squared * 255).astype(np.uint8)

    def configure(self, settings):
        self._otsu_scale_beer = settings["beer_threshold"]
        self._otsu_scale_foam = settings["foam_threshold"]
        self._glass_area = settings["glass_area"]

        self._roi_size = (settings["roi_x1"] - settings["roi_x0"],
                          settings["roi_y1"] - settings["roi_y0"])

        width, height = self._roi_size
        self._roi_center = (settings["roi_x0"] + width // 2,
                          settings["roi_y0"] + height // 2)

    def update(self, image: np.array) -> tuple[float, float, np.array]:
        image = cv2.resize(image, self._frame_size, interpolation=cv2.INTER_LINEAR)

        # using static roi for now
        roi = (self._roi_center, self._roi_size, 0)

        # extract roi
        center, size, angle = roi
        width, height = int(size[0]), int(size[1])
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot, (image.shape[1], image.shape[0]))
        cx, cy = int(center[0]), int(center[1])

        roi_image = rotated_img[cy - height // 2: cy + height // 2, cx - width // 2: cx + width // 2]

        # segment roi
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        hue = cv2.GaussianBlur(hue, (5, 5), 5)
        sat = cv2.GaussianBlur(sat, (5, 5), 5)

        # hue = self.square(hue)
        # sat = self.square(sat)

        beer = cv2.divide(hue, sat, scale=255.0)
        # foam = cv2.divide(sat, hue, scale=255.0)
        foam = cv2.divide(sat, hue, scale=64.0)
        foam = cv2.divide(foam, val, scale=255.0)

        beer_threshold, _ = cv2.threshold(beer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        beer_mask = cv2.inRange(beer, 0, int(self._otsu_scale_beer * beer_threshold))
        beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_OPEN, self._morph_kernel,
                                     iterations=self._morph_open_iterations)
        beer_mask = cv2.morphologyEx(beer_mask, cv2.MORPH_CLOSE, self._morph_kernel,
                                     iterations=self._morph_close_iterations)

        foam_threshold, _ = cv2.threshold(foam, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foam_mask = cv2.inRange(foam, 0, int(self._otsu_scale_foam * foam_threshold))
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, self._morph_kernel,
                                     iterations=self._morph_open_iterations)
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_CLOSE, self._morph_kernel,
                                     iterations=self._morph_close_iterations)

        # compute fill level
        beer_pixels = np.count_nonzero(beer_mask)
        foam_pixels = np.count_nonzero(foam_mask)

        total_pixels = beer_pixels + foam_pixels
        total_fill = total_pixels / self._glass_area
        total_fill = min(total_fill, 1)

        if total_fill > 0.01:
            foam_fill = foam_pixels / (self._glass_area * total_fill)
        else:
            foam_fill = 0

        foam_fill = min(foam_fill, 1)

        total_fill = self._fill_fir.update(total_fill)
        foam_fill = self._foam_fir.update(foam_fill)

        # render debug info
        # box = cv2.boxPoints(roi)
        # box = np.intp(box)
        # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        roi_image[beer_mask == 255] = (0, 0, 255)
        roi_image[foam_mask == 255] = (0, 255, 0)

        cv2.putText(roi_image, f"P {str(total_pixels)}/{str(self._glass_area)}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(roi_image, f"F {str(round(total_fill, 2))}/{str(round(foam_fill, 2))}", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return total_fill, foam_fill, roi_image
