import cv2
import numpy as np
img = cv2.imread("C:\\Users\\naray\\Documents\\kids_python\\sample1.jpg", cv2.IMREAD_GRAYSCALE)
# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)

# Define a threshold for keypoint response
response_threshold = 0.05  # Adjust this value as needed

# Filter keypoints based on their response
filtered_keypoints = [kp for kp in keypoints if kp.response >= response_threshold]

# Extract descriptors corresponding to the filtered keypoints
filtered_descriptors = [descriptors[i] for i, kp in enumerate(keypoints) if kp.response >= response_threshold]
filtered_descriptors = np.array(filtered_descriptors)  # Convert back to NumPy array if needed

# Draw filtered keypoints on the image (for visualization)
img_with_filtered_keypoints = cv2.drawKeypoints(img, filtered_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show or save the result
cv2.imshow('Filtered Keypoints', img_with_filtered_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()