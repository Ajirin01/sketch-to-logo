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

def analyze_image(image_path):
    """
    Analyze image using Google Vision API and get labels.
    """
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return [label.description for label in labels]

def search_image(query, api_key, cse_id, start=1):
    """
    Search for images using Google Custom Search API with pagination support.
    """
    search_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'searchType': 'image',
        'num': 10,  # Number of results per page
        'start': start  # Start index for pagination
    }

    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json()
        items = results.get('items', [])
        return items
    else:
        response.raise_for_status()


@app.route('/upload_and_search', methods=['POST'])
def upload_and_search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Analyze image
        labels = analyze_image(file_path)
        
        # Search for labels using Google Custom Search API
        search_query = ' '.join(labels)
        image_results = search_image(search_query, 'AIzaSyCUJbZgiK2x6S1Lkf_whpazj-41b8WeGxk', '712f655d3c43944fd')
        
        if image_results:
            return jsonify({'images': image_results, 'labels': labels})
        else:
            return jsonify({'message': 'No images found for the query', 'labels': labels}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/search_image', methods=['POST'])
def search_image_route():
    query = request.form.get('query')
    start = int(request.form.get('start', 1))  # Start index for pagination
    if not query:
        return jsonify({'error': 'No search query provided'}), 400

    api_key = 'AIzaSyCUJbZgiK2x6S1Lkf_whpazj-41b8WeGxk'
    cse_id = '712f655d3c43944fd'

    try:
        image_results = search_image(query, api_key, cse_id, start)
        if image_results:
            return jsonify({'images': image_results})
        else:
            return jsonify({'message': 'No images found for the query'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



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
