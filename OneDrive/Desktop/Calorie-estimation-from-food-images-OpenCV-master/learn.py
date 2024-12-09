import os
import numpy as np
import cv2
from create_feature import readFeatureImg  # Ensure this module is in your path
from calorie_calc import getVolume, getCalorie  # Ensure this module is in your path
import csv

# Parameters for SVM
svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383)

def training():
    feature_mat = []
    response = []
    data_path = r"C:\Users\Admin\OneDrive\Desktop\Calorie-estimation-from-food-images-OpenCV-master\Data\images\All_Images"
    
    for j in range(1, 15):
        for i in range(1, 21):
            img_path = os.path.join(data_path, f"{j}_{i}.jpg")
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
            print(f"Processing image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            try:
                fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

            feature_mat.append(fea)
            response.append(float(j))

    trainData = np.float32(feature_mat).reshape(-1, 94)
    responses = np.int32(response)  # Ensure responses are integer for classification

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    print("Training completed and model saved to svm_data.dat")

def testing():
    svm_model = cv2.ml.SVM_load('svm_data.dat')
    feature_mat = []
    response = []
    image_names = []
    pix_cm = []
    fruit_contours = []
    fruit_areas = []
    skin_areas = []

    test_data_path = r"C:\Users\Admin\OneDrive\Desktop\Calorie-estimation-from-food-images-OpenCV-master\Data\images\Test_Images"

    for j in range(1, 15):
        for i in range(21, 26):
            img_path = os.path.join(test_data_path, f"{j}_{i}.jpg")
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
            print(f"Processing image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            try:
                fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)
            response.append([float(j)])
            image_names.append(img_path)

    testData = np.float32(feature_mat).reshape(-1, 94)
    responses = np.float32(response)
    result = svm_model.predict(testData)[1].ravel()
    mask = result == responses.ravel()

    # Open CSV file for writing results
    with open('results.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Image Path', 'Predicted Class', 'Volume', 'Mass', 'Calories', 'Calories per 100g'])

        # Calculate calories and write results to CSV
        for i in range(len(result)):
            volume = getVolume(result[i], fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i])
            mass, cal, cal_100 = getCalorie(result[i], volume)
            csv_writer.writerow([image_names[i], result[i], volume, mass, cal, cal_100])
            print(f"Image: {image_names[i]}, Predicted Class: {result[i]}, Volume: {volume}, Mass: {mass}, Calories: {cal}, Calories per 100g: {cal_100}")

    print("Testing completed and results saved to results.csv")

if __name__ == "__main__":
    print("Starting training...")
    training()
    print("Starting testing...")
    testing()
