import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Density - gram / cm^3
density_dict = {1: 0.609, 2: 0.94, 3: 0.577, 4: 0.641, 5: 1.151, 6: 0.482, 7: 0.513, 8: 0.641, 9: 0.481, 10: 0.641, 11: 0.521, 12: 0.881, 13: 0.228, 14: 0.650}
# kcal per 100 grams
calorie_dict = {1: 52, 2: 89, 3: 92, 4: 41, 5: 360, 6: 47, 7: 40, 8: 158, 9: 18, 10: 16, 11: 50, 12: 61, 13: 31, 14: 30}
# Skin of photo to real multiplier
skin_multiplier = 5 * 2.3

def getCalorie(label, volume):
    '''
    Inputs are the volume of the food item and the label of the food item
    so that the food item can be identified uniquely.
    The calorie content in the given volume of the food item is calculated.
    '''
    try:
        label = int(label)
    except ValueError:
        logging.error(f"Invalid label format: {label}")
        return None, None, None

    if label not in calorie_dict or label not in density_dict:
        logging.error(f"Label {label} not found in calorie or density dictionary.")
        return None, None, None
    
    calorie = calorie_dict[label]
    density = density_dict[label]
    
    if volume is None:
        logging.error("Volume is None.")
        return None, None, calorie
    
    mass = volume * density
    calorie_tot = (calorie / 100.0) * mass
    return mass, calorie_tot, calorie  # calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    '''
    Using calibration techniques, the volume of the food item is calculated using the
    area and contour of the food item by comparing the food item to standard geometric shapes
    '''
    try:
        label = int(label)
    except ValueError:
        logging.error(f"Invalid label format: {label}")
        return None
    
    area_fruit = (area / skin_area) * skin_multiplier  # area in cm^2
    
    if label not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        logging.warning(f"Label {label} not recognized in volume calculation.")
        return None
    
    volume = 100  # Default volume if no specific shape matches
    
    if label in [1, 6, 7, 9, 12]:  # Sphere: apple, tomato, orange, kiwi, onion
        radius = np.sqrt(area_fruit / np.pi)
        volume = (4 / 3) * np.pi * radius**3
        logging.info(f"Label {label} as sphere: radius = {radius}, volume = {volume}")
    
    elif label in [2, 10] or (label == 4 and area_fruit > 30):  # Cylinder: banana, cucumber, carrot
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1]) * pix_to_cm_multiplier
        radius = area_fruit / (2.0 * height)
        volume = np.pi * radius**2 * height
        logging.info(f"Label {label} as cylinder: height = {height}, radius = {radius}, volume = {volume}")
    
    elif (label == 4 and area_fruit < 30) or label in [5, 11]:  # Cheese, sauce
        volume = area_fruit * 0.5  # Assuming width = 0.5 cm
        logging.info(f"Label {label} as rectangular shape: volume = {volume}")

    elif label in [3, 8, 13, 14]:  # Known to have issues or not defined
        volume = None
        logging.info(f"Label {label} has no volume calculation defined.")

    return volume
