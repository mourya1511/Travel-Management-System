import os
import numpy as np
import tensorflow as tf
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.spatial import distance
from create_feature import readFeatureImg
from calorie_calc import getVolume, getCalorie
import csv
import faiss

# Load pre-trained ResNet50 model for feature extraction
def get_model():
    base_model = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return base_model

# Load saved features and other data
def load_saved_data():
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    image_paths = np.load('image_paths.npy')
    scaler = np.load('scaler.npy', allow_pickle=True).item()
    return features, labels, image_paths, scaler

# Build FAISS index
def build_faiss_index(features):
    d = features.shape[1]  # dimension of features
    index = faiss.IndexFlatL2(d)
    index.add(features)
    return index

# Function to find the most similar image in the dataset
def find_most_similar_image(feature, index, dataset_features):
    D, I = index.search(feature, 1)
    return I[0][0]

# Function to test a new image
def test_image(image_path, model, index, scaler, dataset_features, dataset_labels, image_paths):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return

        fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(image_path)
        
        # Preprocess image for ResNet
        img_resized = cv2.resize(img, (224, 224))
        img_array = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_resized, axis=0))

        # Extract features
        features = model.predict(img_array).flatten()

        combined_features = np.hstack([fea, features])
        scaled_feature = scaler.transform([combined_features])
        
        most_similar_index = find_most_similar_image(scaled_feature, index, dataset_features)

        predicted_class = dataset_labels[most_similar_index]
        matched_image_path = image_paths[most_similar_index]

        # Estimate volume and calories based on predicted class
        volume = getVolume(predicted_class, farea, skinarea, pix_to_cm, fcont)
        mass, cal, cal_100 = getCalorie(predicted_class, volume)

        # Print results
        print(f"Uploaded Image: {image_path}")
        print(f"Matched Dataset Image: {matched_image_path}")
        print(f"Predicted Class: {predicted_class}")
        if volume is not None:
            print(f"Estimated Volume: {volume:.2f} cubic centimeters")
        else:
            print("Volume: N/A")
        if mass is not None:
            print(f"Mass: {mass:.2f}")
        else:
            print("Mass: N/A")
        if cal is not None:
            print(f"Calories: {cal:.2f}")
        else:
            print("Calories: N/A")
        if cal_100 is not None:
            print(f"Calories per 100g: {cal_100:.2f}")
        else:
            print("Calories per 100g: N/A")

        # Save results to CSV
        with open('results.csv', mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([image_path, matched_image_path, predicted_class, volume, mass, cal, cal_100])
        print("Results saved to results.csv")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Main function to select and test a new image
def main():
    features, labels, image_paths, scaler = load_saved_data()
    model = get_model()
    index = build_faiss_index(features)

    # Create a Tkinter root window (it will not be shown)
    root = Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image file
    file_path = askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )

    if file_path:
        test_image(file_path, model, index, scaler, features, labels, image_paths)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
