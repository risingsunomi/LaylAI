#!/usr/bin/env python3

"""
test yolo face
"""

from yoloface import face_analysis
import cv2

face=face_analysis()
cap = cv2.VideoCapture(0)
while True: 
    _, frame = cap.read()
    _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    output_frame=face.show_output(frame,box,frame_status=True)
    cv2.imshow('frame',output_frame)
    key=cv2.waitKey(1)
    if key ==ord('v'): 
        break 
cap.release()
cv2.destroyAllWindows()

