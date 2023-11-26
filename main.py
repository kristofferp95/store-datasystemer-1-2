from flask import Flask, render_template, request, redirect
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maks filstørrelse

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_thresholding(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return thresh_image

@app.route('/region-growing', methods=['GET'])
def region_growing():
    return render_template('region_growing.html')

# Enkel Watershed-prosesseringsfunksjon
def apply_watershed(image):
    # Konverter bildet til gråtone
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bruk en binær terskel
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Fjern støy
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Finn bakgrunnsområdet
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finn forgrunnsområdet
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Finn ukjent område
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Merk forgrunnsområdet
    _, markers = cv2.connectedComponents(sure_fg)
    # Legg til 1 til alle merkene slik at sikker bakgrunn ikke er 0, men 1
    markers = markers + 1
    # Merk ukjent område med 0
    markers[unknown == 255] = 0
    # Bruk Watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255,0,0]  # Markere grensene med rødt
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        processing_method = request.form.get('processing_method')
        image_paths = []

        files = request.files.getlist('image')
        if not files:
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(original_file_path)

                image = cv2.imread(original_file_path)
                if processing_method == 'watershed':
                    processed_image = apply_watershed(image)
                else:
                    processed_image = apply_thresholding(image)

                processed_filename = processing_method + '_' + filename
                processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_image_path, processed_image)

                image_paths.append({
                    'original': filename,
                    'processed': processed_filename,
                    'method': processing_method
                })

        return render_template('result.html', image_paths=image_paths)
    
    return render_template('upload.html')



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
