#import numpy as np
#import matplotlib.pyplot as plt
#import cv2
#img = cv2.imread("C:\python31\misc\soccer_ball.jpg", cv2.IMREAD_COLOR)
#plt.imshow(img)
#plt.waitforbuttonpress()
#plt.close('all')

#import cv2, numpy and matplotlib libraries
#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
#img=cv2.imread("geeks.png")
#Displaying image using plt.imshow() method
#plt.imshow(img)

import numpy as np
import cv2
img = cv2.imread("C:\\python31\\misc\\triangle.png")

imgrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgrey, 240, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),True)
    if len(approx) == 3:
        print("triangle")
    if len(approx) == 4:
        print("rectangle or square")


#cv2.imshow("thresh_image",thresh)
cv2.imshow("asd", img)
cv2.waitKey(0)
cv2.destroyAllWindows()