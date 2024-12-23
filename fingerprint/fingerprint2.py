import cv2


img = cv2.imread("C:\\Users\\naray\\Documents\\kids_python\\sample.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:\\Users\\naray\\Documents\\kids_python\\dad2.png", cv2.IMREAD_GRAYSCALE)







sift = cv2.SIFT_create()
keypoints_1, desc_1 = sift.detectAndCompute(img, None)
keypoints_2, desc_2 = sift.detectAndCompute(img2, None)






matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(desc_1, desc_2, k=2)
match_points = []
for p, q in matches:
    
    if p.distance < 0.5 * q.distance:
        match_points.append(p)
        #print(p.distance,q.distance)



print(f"sample: features: {len(keypoints_1)}")
print(f"sample2: features: {len(keypoints_2)}")
print(f"matches: {len(match_points)}")
keypoints = 0
if len(keypoints_1) < len(keypoints_2):
    keypoints = len(keypoints_1)
else:
    keypoints = len(keypoints_2)

score = len(match_points) / keypoints * 100
print(score)

img = cv2.drawKeypoints(img, keypoints_1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
result = cv2.drawMatches(img, keypoints_1, img2, keypoints_2, match_points, None)

cv2.imshow('Image Window', result)
# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()