from flask import Flask, render_template, request, redirect
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # Allowed file extensions
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to apply thresholding to an image
def apply_thresholding(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return thresh_image

# Route for the region-growing page
@app.route('/region-growing', methods=['GET'])
def region_growing():
    return render_template('region_growing.html')

# Simple Watershed processing function
def apply_watershed(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Find the background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Find the foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Find the unknown area
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Label the foreground with different markers
    _, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all markers to ensure that the sure background is not 0, but 1
    markers = markers + 1
    # Mark the unknown area with 0
    markers[unknown == 255] = 0
    # Apply Watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255,0,0]  # Mark boundaries in red
    return image

# Route for the main page for uploading images
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        processing_method = request.form.get('processing_method')  # Get selected processing method from the form

        # List to store image paths for both original and processed images
        image_paths = []

        # Get a list of uploaded files from the form
        files = request.files.getlist('image')

        # Check if any files were uploaded
        if not files:
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):  # Check if the file has an allowed extension
                filename = secure_filename(file.filename)  # Secure the filename to prevent malicious input

                # Save the original uploaded file to a temporary location
                original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(original_file_path)

                # Read the original image using OpenCV
                image = cv2.imread(original_file_path)

                # Apply the selected processing method to the image
                if processing_method == 'watershed':
                    processed_image = apply_watershed(image)
                else:
                    processed_image = apply_thresholding(image)

                # Generate a filename for the processed image based on the selected method
                processed_filename = processing_method + '_' + filename

                # Save the processed image to the upload folder
                processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_image_path, processed_image)

                # Append information about the original and processed images to the image_paths list
                image_paths.append({
                    'original': filename,  # Original image filename
                    'processed': processed_filename,  # Processed image filename
                    'method': processing_method  # Selected processing method
                })

        # Render the result.html template and pass the image_paths list to display the results
        return render_template('result.html', image_paths=image_paths)
    
    # If the request method is GET (initial page load), render the upload.html template
    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
