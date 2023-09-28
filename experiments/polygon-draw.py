import supervision as sv
import numpy as np
import cv2
from pathlib import Path


file = Path('../video/images/2/image-0000300.png')
color = sv.Color.green()
# color = sv.Color.from_hex()

scene = cv2.imread(str(file))
POLYGON = np.array([[75, 300], [215, 300], [360, 400], [160, 420]])

pic = sv.draw_polygon(scene=scene,
                      polygon=POLYGON,
                      color=color,
                      thickness=0)

cv2.imshow('pic', pic)
cv2.waitKey(0)

cv2.destroyWindow('pic')
