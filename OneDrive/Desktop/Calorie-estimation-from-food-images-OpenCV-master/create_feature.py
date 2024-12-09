import cv2
import numpy as np
from feature_moments import getShapeFeatures
from feature_gabor import getTextureFeature
from feature_color import getColorFeature
from img_seg import getAreaOfFood

def createFeature(img):
    '''
    Creates the feature vector of the image using the three features -
    color, texture, and shape features
    '''
    feature = []
    areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = getAreaOfFood(img)
    color = getColorFeature(colourImg)
    texture = getTextureFeature(colourImg)
    shape = getShapeFeatures(binaryImg)
    
    feature.extend(color)
    feature.extend(texture)
    feature.extend(shape)
    
    M = max(feature)
    m = min(feature)
    feature = list(map(lambda x: x * 2, feature))
    feature = [(x - M - m) / (M - m) for x in feature]
    mean = np.mean(feature)
    dev = np.std(feature)
    feature = [(x - mean) / dev for x in feature]

    return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier

def readFeatureImg(filename):
    '''
    Reads an input image when the filename is given,
    and creates the feature vector of the image.
    '''
    img = cv2.imread(filename)
    f, farea, skinarea, fcont, pix_to_cm = createFeature(img)
    return f, farea, skinarea, fcont, pix_to_cm

if __name__ == '__main__':
    import sys
    f = readFeatureImg(sys.argv[1])
    print(f, len(f))
