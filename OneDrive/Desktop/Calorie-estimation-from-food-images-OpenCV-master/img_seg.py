import cv2
import numpy as np

def getAreaOfFood(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, None, None, None, None
    
    # Assuming the largest contour is the food item
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute bounding rectangle and its parameters
    skin_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(skin_rect)
    box = np.int0(box)  # Convert to integer points
    
    # Compute areas and other features
    areaFruit = cv2.contourArea(largest_contour)
    binaryImg = cv2.drawContours(np.zeros_like(gray), [largest_contour], -1, 255, thickness=cv2.FILLED)
    colourImg = cv2.bitwise_and(img, img, mask=binaryImg)
    areaSkin = cv2.contourArea(largest_contour)  # Placeholder, may need adjustment
    fruitContour = largest_contour
    pix_to_cm_multiplier = 1  # Placeholder, adjust as needed

    return areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier
