import os
import io
import numpy as np
import cv2
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the face recognition model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load cached embeddings
def load_embeddings_cache(cache_path="embeddings_cache.npz"):
    data = np.load(cache_path, allow_pickle=True)
    return data["embeddings"], data["names"]

try:
    known_embeddings, known_names = load_embeddings_cache()
    print(f"Loaded {len(known_embeddings)} known embeddings.")
except:
    known_embeddings, known_names = [], []
    print("Failed to load embeddings.")

# Function to convert distance to confidence score
def face_distance_to_confidence(distance, threshold=1.2):
    return round(1.0 - (distance / threshold), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles image uploads, processes them with face recognition, 
    and redirects to the results page with the processed image.
    """
    if 'image' not in request.files:
        return "No file part in request.", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read image
    image = cv2.imread(filepath)
    if image is None:
        return "Error reading image.", 400

    # Detect faces
    faces = face_app.get(image)

    detected_faces = []  # List to store detected names
    
    # Sort faces by x-coordinate (left to right)
    faces = sorted(faces, key=lambda face: face.bbox[0])

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding

        # Default to "Unknown"
        name = "Unknown"
        confidence = 0.0

        # Compare to known embeddings
        if len(known_embeddings) > 0:
            distances = [np.linalg.norm(embedding - k) for k in known_embeddings]
            min_dist_index = int(np.argmin(distances))
            min_dist = distances[min_dist_index]

            name = known_names[min_dist_index]
            confidence = face_distance_to_confidence(min_dist, 1.2)

        # Add face to list with a number
        face_label = f"{i+1}. {name} ({confidence*100:.1f}%)"
        detected_faces.append(face_label)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label with name and confidence
        label = f"{i+1}. {name}"  # Numbered label
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # Draw background rectangle at the bottom of bounding box
        cv2.rectangle(image, (x1, y2), (x1 + label_width + 10, y2 + label_height + 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1 + 5, y2 + label_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Save processed image
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_filepath, image)

    # Redirect to results page with processed image and detected names
    return redirect(url_for('results', image_name=processed_filename, names="|".join(detected_faces)))

@app.route('/results/<image_name>')
def results(image_name):
    """
    Displays the processed image with bounding boxes and labels,
    and shows a numbered list of detected faces.
    """
    image_url = url_for('static', filename=f"uploads/{image_name}")
    names_str = request.args.get("names", "")
    detected_faces = names_str.split("|") if names_str else []

    return render_template('results.html', image_url=image_url, detected_faces=detected_faces)

class Individual:
    def __init__(self, name, number):
        self.name = name
        self.number = number

if __name__ == '__main__':
    app.run(debug=True)
