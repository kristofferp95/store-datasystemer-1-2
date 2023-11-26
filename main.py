from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
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

@app.route('/threshold', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('image')
        if not files:
            return redirect(request.url)

        original_images = []
        thresholded_images = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                image = cv2.imread(file_path)
                thresholded_image = apply_thresholding(image)

                original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
                thresholded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thresholded_' + filename)
                cv2.imwrite(original_image_path, image)
                cv2.imwrite(thresholded_image_path, thresholded_image)

                original_images.append(original_image_path)
                thresholded_images.append(thresholded_image_path)

        pairs = zip(original_images, thresholded_images)
        return render_template('result.html', pairs=pairs)
    
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
