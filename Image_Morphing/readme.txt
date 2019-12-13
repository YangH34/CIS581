!!! output .avi videos are stored in the folder 'videos'!

!!! Please only use click correspondences to define the control points. Trying to hard input array of control points
    to morph_tri.py will not get what you want due to resize!

!!! Input images are resized to (300, 300, 3) and hence output images are of the same size.
    This has caused the test_script.py to fail running but the program works fine following instructions below!

!!! program will only support up to 99 frames. Above that will cause the problems in video generation due to indexing

-My script is 'owntest.py'.
-The way to run is to input the names of the images user wants to morph in the lines:
    startImage = '(insert start image)'
    endImage = '(insert end image)'
-And then enter the frames and fps if you'd like something different than 36 & 18 in the lines:
    frames = 36
    fps = 18
-A new folder with the name of "startImage + '_to_' + endImage" will be created. Result images of morph will be put in here.
-Note that the images being morphed must be in the same directory as owntest.py.
-The result array of images is resultImageArr. It will be of size (i, 300, 300, 3) where i is the
    number of images generated.
-The result video "startImage + '_to_' + endImage + _video.avi" will be put into the 'videos' folder
-Note that we cannot perform another morph of the same two images unless we delete the old generated folder.