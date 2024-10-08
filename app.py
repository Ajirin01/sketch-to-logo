from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import requests
from google.cloud import vision
import io

# Initialize Flask application
app = Flask(__name__)

# Set upload and output directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/logo_output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def trace_sketch(image_path, lower_threshold=50, upper_threshold=150):
    """
    Trace the uploaded sketch with customizable threshold values.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error reading image at {image_path}")

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, lower_threshold, upper_threshold)

    # Optionally refine the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    inverted_edges = cv2.bitwise_not(edges)
    traced_image = np.full_like(inverted_edges, 255)
    traced_image = cv2.bitwise_and(traced_image, inverted_edges)

    return traced_image

def smooth_contours(image):
    """
    Smooth contours on the image.
    """
    # Convert image to binary
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    smoothed_image = np.zeros_like(image)

    for contour in contours:
        smooth_contour_pts = smooth_contour(contour)
        if len(smooth_contour_pts) > 1:
            cv2.drawContours(smoothed_image, [smooth_contour_pts], -1, (255, 255, 255), thickness=cv2.FILLED)

    return smoothed_image

def smooth_contour(contour, epsilon=0.02):
    """
    Smooth the contour using the approxPolyDP method.
    """
    contour = np.squeeze(contour)
    if len(contour) < 3:
        return contour

    # Approximate the contour with a precision factor
    epsilon = epsilon * cv2.arcLength(contour, True)
    smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
    return smooth_contour

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/preview/<path:image_path>', methods=['GET'])
def preview(image_path):
    """
    Render the index.html template and pass the image path for preview.
    """
    return render_template('index.html', image_path=image_path)

@app.route('/preview_trace', methods=['POST'])
def preview_trace():
    file = request.files.get('file')
    lower_threshold = int(request.form.get('lower_threshold', 50))
    upper_threshold = int(request.form.get('upper_threshold', 150))

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Trace the sketch with dynamic thresholds
    traced_image = trace_sketch(file_path, lower_threshold, upper_threshold)

    print("traced_image", traced_image)
    # Apply smoothing to the traced image
    smoothed_image = smooth_contours(traced_image)
    print("smoothed_image", smoothed_image)

    # Save the smoothed image temporarily for preview
    preview_path = os.path.join(app.config['OUTPUT_FOLDER'], f'preview_{filename}')
    cv2.imwrite(preview_path, traced_image)

    return jsonify({'image_url': f'/static/logo_output/preview_{filename}'})


# def preview_trace():
#     file = request.files.get('file')
#     lower_threshold = int(request.form.get('lower_threshold', 50))
#     upper_threshold = int(request.form.get('upper_threshold', 150))

#     if not file:
#         return jsonify({'error': 'No file uploaded'}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     # Trace the sketch with dynamic thresholds
#     traced_image = trace_sketch(file_path, lower_threshold, upper_threshold)

#     # Save the traced image temporarily for preview
#     preview_path = os.path.join(app.config['OUTPUT_FOLDER'], f'preview_{filename}')
#     cv2.imwrite(preview_path, traced_image)

#     return jsonify({'image_url': f'/static/logo_output/preview_{filename}'})


@app.route('/generate_logo', methods=['POST'])
def generate_logo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Trace the sketch image
    traced_image = trace_sketch(file_path)

    # Save the traced image
    traced_path = os.path.join(app.config['OUTPUT_FOLDER'], f'traced_{filename}')
    cv2.imwrite(traced_path, traced_image)

    # Convert the traced image to SVG
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'logo_{filename}.svg')
    vectorize_trace(traced_path, output_path)

    return render_template('preview.html', image_path=f'logo_output/logo_{filename}.svg')

if __name__ == "__main__":
    app.run(debug=True)
