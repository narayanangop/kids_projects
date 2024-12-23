import cv2

print("hello world")
img = cv2.imread("C:\\Users\\naray\\Documents\\kids_python\\sample1.jpg", cv2.IMREAD_GRAYSCALE)


sift = cv2.SIFT_create()
keypoints = sift.detect(img, None)
img = cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Image Window', img)
# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()