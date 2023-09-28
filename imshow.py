import cv2
import numpy as np
from pathlib import Path

ZONE_IN_POLYGONS = [
    np.array([[75, 300], [215, 300], [360, 400], [160, 420]]),
    np.array([[870, 30], [1215, 320], [1340, 430], [1050, 470]]),
    np.array([[820, 700], [1170, 600], [1520, 760], [1130, 890]]),
    np.array([[0, 460], [100, 440], [255, 610], [0, 620]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[370, 390], [245, 300], [410, 240], [600, 320]]),
    np.array([[1050, 420], [1260, 370], [1370, 455], [1165, 480]]),
    np.array([[], [930, 700], [1165, 850], [880, 935]]),
    np.array([[0, 430], [150, 420], [180, 540], [0, 550]]),
]

pic = Path('video/images/2/image-0000225.png')
ara = cv2.imread(str(pic))
cv2.imshow('win', ara)
cv2.waitKey(0)
cv2.destroyWindow()