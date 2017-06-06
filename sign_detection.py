# Author: Jose Miguel Buenaposada 2015.
# Simple MSER application for traffic sign window proposal generation
import cv2
import numpy as np
import matplotlib as plt
import os
from signals_utils import *
import winsound

TEST = False

SU = SignalsUtils()

if TEST:
    X, Y = SU.read_images(save=True)
    best_clf = SU.test_clfs(save=True)
    winsound.Beep(2500, 500)
else:
    best_clf = SU.load_model()

mser = cv2.MSER_create()

test_dir = 'ImgsAlumnos/test/'
test_ext = ('.jpg', '.ppm')

for filename in os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in test_ext:
        print "Test, processing ", filename, "\n"
        full_path = os.path.join(test_dir, filename)
        I = cv2.imread(full_path)
        Icopy = I.copy()
        Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        regions = mser.detectRegions(I, None)
        rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]

        for r in rects:
            x, y, w, h = r
            # Simple aspect ratio filtering
            aratio = float(w) / float(h)
            if (aratio > 1.2) or (aratio < 0.8):
                continue

            Icrop = I[y:y+h, x:x+w]
            prediction, class_name = SU.predict(Icrop)
            cv2.rectangle(Icopy, (x, y), (x + w, y + h), (0, 255, 0) if prediction == 0 else (0, 0, 255), 2)
            if prediction != 0:
                cv2.putText(Icopy, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow('img', Icopy)
        key = cv2.waitKey(0)

        if key == ord('q') & 0xff:
            break
