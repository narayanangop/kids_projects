import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance

def extract_minutiae(image):
    """
    Extracts minutiae points (ridge endings and bifurcations) from a fingerprint image.
    """
    # Skeletonize the binary image to reduce ridges to one-pixel width
    skeleton = skeletonize(image // 255).astype(np.uint8)
    
    # Identify minutiae points (endpoints and bifurcations)
    minutiae = []
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 1:
                # Count the number of 1's in the 3x3 neighborhood
                neighbors = skeleton[i-1:i+2, j-1:j+2]
                count = np.sum(neighbors) - 1  # Subtract the center pixel
                
                if count == 1:  # Ridge ending
                    minutiae.append((i, j, 'ending'))
                elif count > 2:  # Bifurcation
                    minutiae.append((i, j, 'bifurcation'))
    return minutiae

def match_minutiae(minutiae1, minutiae2, threshold=15):
    """
    Matches minutiae points from two sets with a one-to-one correspondence.
    """
    matches = []
    used_minutiae2 = set()  # Keep track of matched minutiae in image 2

    for m1 in minutiae1:
        best_match = None
        best_distance = float('inf')

        for i, m2 in enumerate(minutiae2):
            if i in used_minutiae2:  # Skip already matched minutiae
                continue
            
            # Compute the Euclidean distance between minutiae points
            dist = distance.euclidean(m1[:2], m2[:2])

            if dist <= threshold and dist < best_distance and m1[2] == m2[2]:  # Match type and distance
                best_match = (m1, m2)
                best_distance = dist
        
        if best_match:
            matches.append(best_match)
            used_minutiae2.add(minutiae2.index(best_match[1]))  # Mark this minutiae as used

    return matches


# Load and preprocess the first fingerprint image
img1 = cv2.imread('C:\\Users\\naray\\Documents\\kids_python\\sample.png', cv2.IMREAD_GRAYSCALE)
_, binary1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)

# Load and preprocess the second fingerprint image
img2 = cv2.imread('C:\\Users\\naray\\Documents\\kids_python\\dad2.png', cv2.IMREAD_GRAYSCALE)
_, binary2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)

# Extract minutiae points from both images
minutiae1 = extract_minutiae(binary1)
minutiae2 = extract_minutiae(binary2)

# Match minutiae points
matches = match_minutiae(minutiae1, minutiae2)

# Display results
print(f"Number of minutiae in Image 1: {len(minutiae1)}")
print(f"Number of minutiae in Image 2: {len(minutiae2)}")
print(f"Number of matches: {len(matches)}")

# Ensure img2 matches img1 in height and type
if img1.shape[0] != img2.shape[0] or img1.dtype != img2.dtype:
    img2 = cv2.resize(img2, (img2.shape[1], img1.shape[0]))
    img2 = img2.astype(img1.dtype)

# Concatenate horizontally
img_combined = cv2.hconcat([img1, img2])

# Visualize matched minutiae
for match in matches:
    (x1, y1, _), (x2, y2, _) = match
    cv2.circle(img_combined, (y1, x1), 5, (0, 255, 0), -1)
    cv2.circle(img_combined, (y2 + img1.shape[1], x2), 5, (255, 0, 0), -1)
    cv2.line(img_combined, (y1, x1), (y2 + img1.shape[1], x2), (0, 255, 255), 1)

cv2.imshow('Matched Minutiae', img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
