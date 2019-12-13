import numpy as np
import cv2
from interp import interp2
from getFeatures import pointsToBox

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


# get a cropped 2D image from @original, searchsize is the length of the box for cropping
# @rowInd and @colInd are the center of the box
def getSearchZone(original, colInd, rowInd, searchsize):
    H,W = np.shape(original)
    if (rowInd >= H or colInd >= W):
        print("invalid coordinates")
        print(H, W)
        print("colInd = ", colInd)
        print("rowInd = ", rowInd)
        return
    radius = int(searchsize / 2)
    rowlow = max(0, rowInd-radius)
    rowhigh = min(H - 1, rowInd+radius)
    collow = max(0, colInd-radius)
    colhigh = min(W - 1, colInd+radius)
    return original[int(rowlow):int(rowhigh), int(collow):int(colhigh)]


def leastSquareDisplacement(im_before, im_after):
    im_before_edges_x = cv2.Sobel(im_before, cv2.CV_64F, 1, 0)
    im_before_edges_y = cv2.Sobel(im_before, cv2.CV_64F, 0, 1)
    It = im_after.ravel() - im_before.ravel()

    im_before_edges_x = im_before_edges_x.ravel()
    im_before_edges_y = im_before_edges_y.ravel()

    IxIx = np.inner(im_before_edges_x,im_before_edges_x)
    IyIy = np.inner(im_before_edges_y,im_before_edges_y)
    IxIy = np.inner(im_before_edges_x, im_before_edges_y)
    IxIt = np.inner(im_before_edges_x, It)
    IyIt = np.inner(im_before_edges_y, It)

    AtA = np.array([[IxIx, IxIy], [IxIy, IyIy]])
    Atb = np.array([[IxIt], [IyIt]])
    disp = np.zeros((2,1))
    try:
        disp = np.linalg.solve(AtA, -Atb)
    except np.linalg.LinAlgError:
        print("singular matrix")
        print("shape of AtA is ", np.shape(AtA))
        print("shape of Atb is ", np.shape(Atb))
    return disp


def displaceIm(im, disp):
    H, W = np.shape(im)
    colIndex, rowIndex = np.meshgrid(range(W), range(H))
    colIndex = colIndex + np.ones((H,W),np.float64) * disp[0]
    rowIndex = rowIndex + np.ones((H,W),np.float64) * disp[1]
    return interp2(im, colIndex, rowIndex)

def iterSolveDisplacement(im_before, im_after, iter):
    im_curr = im_before
    disp_total = np.array([[0],[0]])
    for i in range(iter):
        # print("iteration ", i)
        displacement = leastSquareDisplacement(im_curr, im_after)
        if (displacement[0] == 0 and displacement[1] == 0):
            print("zero displacement")
            return im_curr, disp_total
        disp_total = disp_total - displacement
        im_next = displaceIm(im_before, disp_total)
        im_curr = im_next
        # print(calculateError(im_curr, im_after))
    return im_curr, disp_total



# assumption: pixel movement between frame is less than searchsize
# disp[0] is the offset in x; disp[1] is the offset in y
def estimateFeatureTranslation(startX, startY, im_before, im_after):
    startX = float(startX)
    startY = float(startY)
    # print(startX, startY)
    searchsize = 16
    cropped_before = getSearchZone(im_before, startX, startY, searchsize)
    cropped_after = getSearchZone(im_after, startX, startY, searchsize)

    cropped_before = cv2.GaussianBlur(cropped_before, (5, 5), cv2.BORDER_DEFAULT)
    cropped_after = cv2.GaussianBlur(cropped_after, (5, 5), cv2.BORDER_DEFAULT)


    if (np.shape(cropped_before) != (searchsize, searchsize)):
        print("bad coordinate", startX, startY)
        disp = np.array([[10000], [10000]])
        return disp[0], disp[1]

    _, disp = iterSolveDisplacement(cropped_before, cropped_after, 30)
    if (disp[0] > 10 or disp[1] > 10):
        print("too much displacement")
    return disp[0], disp[1]



def writeBox(feats_x_prev, feats_y_prev, frame_prev, frame_curr, frame_curr_color):
    H, W = np.shape(frame_curr)
    feats_x_curr = np.empty((0,1), feats_x_prev.dtype)
    feats_y_curr = np.empty((0,1), feats_y_prev.dtype)
    for i in range(len(feats_x_prev)):
        delta_x, delta_y = estimateFeatureTranslation(feats_x_prev[i][0], feats_y_prev[i][0], frame_prev, frame_curr)
        try_x = feats_x_prev[i][0] - delta_x
        try_y = feats_y_prev[i][0] - delta_y
        if (try_x < W and try_y < H and try_x >= 0 and try_y >= 0):
            feats_x_curr = np.append(feats_x_curr, np.array([try_x]), axis=0)
            feats_y_curr = np.append(feats_y_curr, np.array([try_y]), axis=0)

    frame_curr_with_box = pointsToBox(frame_curr_color, np.hstack((feats_x_prev, feats_y_prev)))
    return feats_x_curr, feats_y_curr



