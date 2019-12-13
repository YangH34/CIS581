import numpy as np
import cv2
import os
from getFeatures import interactiveGetFeats
from getFeatures import pointsToBox
from script import video_create
from main_script import writeBox


frame0 = cv2.imread('frame0.jpg')
frame0_color = frame0
frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

frame1 = cv2.imread('frame1.jpg')
frame1_color = frame1
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


med_frame0 = cv2.imread('medium_frame0.jpg')
med_frame_color = med_frame0
med_frame0 = cv2.cvtColor(med_frame0, cv2.COLOR_BGR2GRAY)

med_frame1 = cv2.imread('medium_frame1.jpg')
med_frame1_color = med_frame1
med_frame1 = cv2.cvtColor(med_frame1, cv2.COLOR_BGR2GRAY)

hard_frame0 = cv2.imread('hard_frame0.jpg')
hard_frame0 = cv2.cvtColor(hard_frame0, cv2.COLOR_BGR2GRAY)

hard_frame1 = cv2.imread('hard_frame1.jpg')
hard_frame1 = cv2.cvtColor(hard_frame1, cv2.COLOR_BGR2GRAY)



# #script starts here
feats_x_prev, feats_y_prev, frame0_with_box = interactiveGetFeats(frame0)
# feats_x_prev_b, feats_y_prev_b, frame0_with_box = interactiveGetFeats(frame0)

feats_x_curr, feats_y_curr = writeBox(feats_x_prev, feats_y_prev, frame0, frame1, frame1_color)
# feats_x_curr_b, feats_y_curr_b = writeBox(feats_x_prev_b, feats_y_prev_b, frame0, frame1, frame1_color)

frame_curr_with_box = pointsToBox(frame1_color, np.hstack((feats_x_prev, feats_y_prev)))
# frame_curr_with_box = pointsToBox(frame_curr_with_box, np.hstack((feats_x_prev_b, feats_y_prev_b)))


#-----------------------------------------------------------------------------------------
parent_dir = os.getcwd()
# take a video and grab the frames you want for testing
cap = cv2.VideoCapture('video_easy.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

prev_frame = frame0
frame_num = 0
count = 100


while(True):
    # Capture frame-by-frame
    ret, curr_frame = cap.read()
    print("processing frame ", frame_num)
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break
    if (frame_num == 0):
        frame_num += 1
        print("skipping frame0")
        continue
    curr_frame_color = curr_frame
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # Saves image of the current frame in jpg file
    name = 'trackingframe' + str(frame_num) + '.jpg'
    # print ('Creating...' + name)
    feats_x_curr, feats_y_curr = writeBox(feats_x_prev, feats_y_prev, prev_frame, curr_frame, curr_frame_color)
    # feats_x_curr_b, feats_y_curr_b = writeBox(feats_x_prev_b, feats_y_prev_b, prev_frame, curr_frame, curr_frame_color)
    frame_curr_with_box = pointsToBox(curr_frame_color, np.hstack((feats_x_prev, feats_y_prev)))
    # frame_curr_with_box = pointsToBox(frame_curr_with_box, np.hstack((feats_x_prev_b, feats_y_prev_b)))

    # frame_curr_with_box = np.rot90(frame_curr_with_box, 3)
    cv2.imwrite(parent_dir + '/out_frames/out' + str(frame_num + count) + '.jpg', frame_curr_with_box)
    prev_frame = curr_frame
    feats_x_prev = feats_x_curr
    feats_y_prev = feats_y_curr
    # feats_x_prev_b = feats_x_curr_b
    # feats_y_prev_b = feats_y_curr_b
    frame_num += 1

cap.release()
cv2.destroyAllWindows()


Image = 'testvid'
fps = 8
nr = 3
nc = 3
directory = 'out_frames'
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory)
video_create(path, Image, parent_dir, fps)