from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from create_feature import readFeatureImg
from calorie_calc import getVolume, getCalorie

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def test_image(image_path):
    """Process the image and return prediction results."""
    if not os.path.exists(image_path):
        return "File not found."

    svm_model = cv2.ml.SVM_load('svm_data.dat')

    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Failed to read image."

        fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(image_path)
        feature_mat = np.float32([fea]).reshape(-1, 94)
        result = svm_model.predict(feature_mat)[1].ravel()

        if result.size == 0:
            return "Prediction result is empty."

        predicted_class = result[0]
        volume = getVolume(predicted_class, farea, skinarea, pix_to_cm, fcont)
        mass, cal, cal_100 = getCalorie(predicted_class, volume)

        results = {
            "Predicted Class": predicted_class,
            "Volume": volume if volume is not None else "N/A",
            "Calories": cal if cal is not None else "N/A"
        }
        return results

    except Exception as e:
        return f"Error processing image: {e}"

@app.route('/')
def upload_form():
    """Render the upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and processing."""
    if 'image' not in request.files:
        return "No image part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = test_image(file_path)
        if isinstance(results, dict):
            return render_template('results.html', results=results)
        else:
            return results

    return "File type not allowed"

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
