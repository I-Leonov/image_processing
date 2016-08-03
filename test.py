from transform import four_point_transform
import imutils
from skimage.filters import threshold_adaptive
import argparse
import numpy as np
import cv2
import pyocr.builders
from PIL import Image as PI

def parse(img):
    reduced = cv2.reduce(img, 1, cv2.REDUCE_AVG)

    y = 0
    count = 0
    isSpace = 0
    ycoords = []

    for i in range(0, img.shape[0]):
        if isSpace == 0:
            if (reduced.item(i) > 250):
                isSpace = 1
                count = 1
                y = i
        else:
            if (reduced.item(i) < 250):
                isSpace = 0
                ycoords.append(y / count)
            else:
                y += i
                count += 1
    tool = pyocr.get_available_tools()[0]
    #
    x1 = 0
    x2 = img.shape[1]
    for i in range(0, len(ycoords)):

        if i != 0:
            y1 = ycoords[i - 1]
        else:
            y1 = 0
        y2 = ycoords[i]

        txt = tool.image_to_string(
            # PI.open("warped_line.png"),
            PI.fromarray(img[y1:y2, x1:x2]),
            lang="eng",
            builder=pyocr.builders.TextBuilder()
        )
        print txt

    y1 = ycoords[len(ycoords) - 1]
    y2 = warped.shape[0]

    txt = tool.image_to_string(
        # PI.open("warped_line.png"),
        PI.fromarray(img[y1:y2, x1:x2]),
        lang="eng",
        builder=pyocr.builders.TextBuilder()
    )
    print txt


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])

ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print "STEP 1: Edge Detection"
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
try: screenCnt
except NameError:
    warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 251, offset=10)
    warped = warped.astype("uint8") * 255
    # warped = (255-warped)
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (10, 10))
    warped = cv2.erode(warped, kernel)
    cv2.imshow("Scanned11", warped)
    cv2.waitKey(0)
    #
    pts = cv2.findNonZero(warped)
    box = cv2.minAreaRect(pts)
    # # if (box[1][0] > box[1][1]):
    if (1):
        points = cv2.boxPoints(box)
        tst = np.int0(points)
        cv2.drawContours(image, [tst], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for  i in range (0,4):
            cv2.line(warped, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0))
        M = cv2.getRotationMatrix2D(box[0], box[2], 1.0)
        warped = cv2.warpAffine(warped, M, (warped.shape[0],warped.shape[1]))

else:

    # show the contour (outline) of the piece of paper
    print "STEP 2: Find contours of paper"
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 251, offset=10)
    warped = warped.astype("uint8") * 255


# show the original and scanned images
print "STEP 3: Apply perspective transform"

cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)

# reduced=np.array([])
parse(warped)
print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
parse(image)

cv2.imwrite("warped.jpg", warped)

