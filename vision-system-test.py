import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

#biblioteki do raspberry kamera
#from picamera.array import PiRGBArray
#from picamera import PiCamera

""""
WYKRYWANIE OBIEKTÓW
"""
class HomogeneousBgDetector():
    def __init__(self):
        pass
    def detect_objects(self, frame):
        # zmiAna na skale szarosci
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold Mask
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # kontury
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow(mask)
        objects_contours = []

        #spis punktów w konturach
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)

"""=================================================================="""


""""
WYKRYWANIE KOLORU
"""
class ColorDetection:
    def __init__(self):
        colors = OrderedDict({
        "czerwony": (255, 0, 0),
        "zielony": (0, 255, 0),
        "niebieski": (0, 0, 255)})

        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, img, c):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(img, mask=mask)[:3]
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)
            return self.colorNames[minDist[1]]

"""=================================================================="""


#obiekt do wykrywania
detector = HomogeneousBgDetector()

#dodanie kamery dla raspberry może być inne
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while True:
    _, img = cap.read()

    blurred = cv2.GaussianBlur(img, (25, 25), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    contures = detector.detect_objects(img)
    cd = ColorDetection()
    # wyswietla kontury
    y1, x1, z1 = img.shape
    #print(x1, y1, z1)

    h1 = y1 // 2
    w1 = x1 // 2
    #wyswitla srodki krawedzi
    cv2.circle(img, (w1, 0), 5, (255, 0, 0), -1)
    cv2.circle(img, (0, h1), 5, (0, 0, 255), -1)
    print(w1, h1)
    # wyzancza krawedzie
    for cnt in contures:
        # cv2.polylines(img, [cnt], True, (255, 0, 0), 3)
        rectangular = cv2.minAreaRect(cnt) # wyznacza prostokont
        (x, y), (w, h), angle = rectangular
        # print(x, y)
        box = cv2.boxPoints(rectangular)
        box = np.int0(box)
        color = cd.label(lab, cnt) # znajduje kolor
        # print(color)
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.polylines(img, [box], True, (0, 200, 255), 3)
        cv2.putText(img, "X {}".format(round(x, 1)), (int(x - 40), int(y - 50)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv2.putText(img, "Y {}".format(round(y, 1)), (int(x - 40), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv2.putText(img, "Kolor {}".format(color), (int(x - 60), int(y + 20)), cv2.FONT_HERSHEY_PLAIN, 1, (50, 10, 80), 2)

    #linie do srodka przedmiotu
    # cv2.line(img, (0, h1), (int(x),int(y)), (0,200, 200), 2)
    # cv2.line(img, (w1, 0), (int(x),int(y)), (0,200, 200), 2)

    cv2.imshow(img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
