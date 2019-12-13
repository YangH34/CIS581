import numpy as np
import cv2
import os



cap = cv2.VideoCapture('video_hard.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

frame_num = 0

while(frame_num <= 5):
    # Capture frame-by-frame
    ret, curr_frame = cap.read()
    print("processing frame ", frame_num)
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # Saves image of the current frame in jpg file
    name = 'hard_frame' + str(frame_num) + '.jpg'
    cv2.imwrite(name, curr_frame)
    frame_num += 1

cap.release()
cv2.destroyAllWindows()