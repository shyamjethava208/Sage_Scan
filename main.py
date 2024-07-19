import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import random
# from scan_sage import Scan_sage  # Import your Scan_sage class from scan_sage.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# scan = Scan_sage()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded image
            processed_data = process_image(filepath)

            # Pass processed data to template for display
            return render_template('index.html', filename=filename, processed_data=processed_data)
        else:
            return render_template('index.html', error='File type not allowed')

    return render_template('index.html')

def process_image(filepath):
    # Read and preprocess the image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0

    # Call Scan_sage methods to process the image
    # scan.process_image(image)  # Modify Scan_sage to handle image processing

    # Example: Get results from Scan_sage class if needed
    # results = scan.get_results()
    result = random.randint(0,2)
    if result == 1:
        return "you have pneumonia"
    else:
        return "you are safe"

    # Return processed data or results as needed
    return "Processed successfully"

if __name__ == '__main__':
    app.run(debug=True)
