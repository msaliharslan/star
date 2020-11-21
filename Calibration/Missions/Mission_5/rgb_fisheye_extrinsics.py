import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

if(os.getcwd().split('/')[-1] != 'star'):
    os.chdir("../../../")
    
# read file names
fileNamesL = glob.glob("SnapShots/t265_d435i/1_2020-11-03_20:56/leftFisheye/*" )
fileNamesR = glob.glob("SnapShots/t265_d435i/1_2020-11-03_20:56/rightFisheye/*")
fileNamesRgb = glob.glob("SnapShots/t265_d435i/1_2020-11-03_20:56/rgb/*")

fileNamesL.sort()
fileNamesR.sort()
fileNamesRgb.sort()

# read images
leftFishImgs = []
rightFishImgs = []
rgbImgs = []

for iml, imr, imrgb in zip(fileNamesL, fileNamesR, fileNamesRgb):
    iml = cv2.imread(iml, cv2.IMREAD_UNCHANGED)
    imr = cv2.imread(imr, cv2.IMREAD_UNCHANGED)
    imrgb = cv2.imread(imrgb, cv2.IMREAD_GRAYSCALE)
    
    leftFishImgs.append(iml)
    rightFishImgs.append(imr)
    rgbImgs.append(imrgb)
    
# undistort fisheye images
K1 = np.array([[284.501708984375, 0.0, 430.9294128417969], [0.0, 285.4164123535156, 394.66510009765625], [0.0, 0.0, 1.0]])
D1 = np.array([-0.00012164260260760784, 0.03437558934092522, -0.03252582997083664, 0.004925379063934088])

K2 = np.array([[284.1828918457031, 0.0, 427.9779052734375], [0.0, 285.0440979003906, 399.5506896972656], [0.0, 0.0, 1.0]])
D2 = np.array([0.0009760634857229888, 0.030147459357976913, -0.02769969031214714, 0.0031066760420799255])

fisheyeWidth = 800
fisheyeHeight = 848

KRGB = np.array([923.2835693359375, 0.0, 639.9102783203125, 0.0, 923.6146240234375, 370.2297668457031, 0.0, 0.0, 1.0])
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)

boardWidth = 8
boardHeight = 8
CHECKERBOARD = (boardHeight, boardWidth)
squareSize = 0.033 # 3.3 cm

def undistortFisheyeImages(fisheyeImages, K, D):
    
    
    # We calculate the undistorted focal length:
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (np.pi/180)  # 90 degree desired fov
    stereo_height_px = 1000          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / np.tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R = np.eye(3)

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])    
    
    undistortedImages = []
    
    nk = K.copy()

    # nk[0,0]=K[0,0]/2
    # nk[1,1]=K[1,1]/2   
    
    for image in fisheyeImages:
            
        undistorted = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=P, new_size=(int(stereo_height_px), int(stereo_width_px)))
        undistortedImages.append(undistorted)

    return undistortedImages, np.array(P[:,:-1])


leftUndistorted, KL = undistortFisheyeImages(leftFishImgs, K1, D1)
rightUndistorted, KR = undistortFisheyeImages(rightFishImgs, K2, D2)


objp = np.zeros((8*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2)
objp = objp * squareSize

imagePointsLeft =  []
imagePointsRgb =  [] 
imgsRgbWP = []
imgsLWP = []

counter = 0
imageCount = len(leftUndistorted)
for i in range(imageCount):

    imageLeft = leftUndistorted[i]
    imageRgb = rgbImgs[i]

    retL, cornersL = cv2.findChessboardCornersSB(imageLeft, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)
    retRgb, cornersRgb = cv2.findChessboardCornersSB(imageRgb, CHECKERBOARD, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if(retL and retRgb):
        cv2.cornerSubPix(imageLeft, cornersL, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.cornerSubPix(imageRgb, cornersRgb, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imagePointsLeft.append(cornersL)
        imagePointsRgb.append(cornersRgb)
        imgsLWP.append(cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB))
        imgsRgbWP.append(cv2.cvtColor(imageRgb, cv2.COLOR_GRAY2RGB))

N_OK = len(imagePointsLeft)

objp = np.array([objp]*len(imagePointsLeft), dtype=np.float32)
objp = np.reshape(objp, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objp, imagePointsLeft, imagePointsRgb, KL, D, KRGB, D, (0,0), flags=cv2.CALIB_FIX_INTRINSIC)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = tuple(map(tuple, pt1))[0]
        pt2 = tuple(map(tuple, pt2))[0]
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,pt1,5,color,-1)
        img2 = cv2.circle(img2,pt2,5,color,-1)
    return img1,img2


errs = []
for i in range(N_OK):
    lines1 = cv2.computeCorrespondEpilines(imagePointsRgb[i].reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(imgsLWP[i], imgsRgbWP[i],lines1, imagePointsLeft[i], imagePointsRgb[i])
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(imagePointsLeft[i].reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(imgsRgbWP[i], imgsLWP[i],lines2, imagePointsRgb[i], imagePointsLeft[i])
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    
    # img3 = cv2.resize(img3, img5.shape)
    # img = cv2.hconcat([img5, img3])
    h1, w1 = img3.shape[:2]
    h2, w2 = img5.shape[:2]

    #create empty matrix
    img = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    #combine 2 images
    img[:h1, :w1,:3] = img3
    img[:h2, w1:w1+w2,:3] = img5
    cv2.imwrite("./temp/"+"George_Bush"+ str(i) +".png", img)
    
    
    ####################################
    ## Error calculation
    ######################
    
    err = 0
    
    for pt, l in zip(imagePointsLeft[i], lines1):
        err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
    
    for pt, l in zip(imagePointsRgb[i], lines2):
        err += abs(pt[0,0] * l[0] + pt[0,1] * l[1] + l[2])
        
        
    err /= lines1.shape[0] + lines2.shape[0]
    errs.append(err)






















