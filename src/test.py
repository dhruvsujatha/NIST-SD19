import numpy as np
import cv2
from imutils.contours import sort_contours
from matplotlib import pyplot as plt
import imutils
from PIL import Image as img

gray = cv2.imread("C:/NN/NIST SD 19/Words/Test 0.jpg", 0)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 100, 200)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]


chars = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)
        else:
            thresh = imutils.resize(thresh, height=32)

        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (50, 50))
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        padded = cv2.resize(padded, (50, 50))
        cv2.imshow("image", padded)
        cv2.waitKey(0)
        print(np.shape(np.asarray(padded)))
        chars.append((padded, (x, y, w, h)))
