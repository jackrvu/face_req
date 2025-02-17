# Face Recognition Web Application

This web application provides face detection and recognition capabilities through a simple web interface. It uses InsightFace for face detection and recognition, Flask for the web server, and OpenCV for image processing.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/yourusername/face_req_site.git
   cd face_req_site
   ```

2. **Install required Python packages**:
   ```bash
   pip install flask numpy opencv-python insightface pillow werkzeug
   ```

3. **Create required directories**:
   The application will automatically create a `static/uploads` directory for storing uploaded images.

## Usage

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://localhost:5000`

3. **Upload an image**:
   - Click the upload button to select an image containing faces
   - The application will process the image and display results showing:
     - Detected faces with bounding boxes
     - Numbered labels for each face
     - Recognition confidence scores (if known faces are loaded)

## Features

- **Face Detection**: Automatically detects faces in uploaded images
- **Face Recognition**: Compares detected faces against known embeddings
- **Confidence Scoring**: Displays confidence scores for recognized faces
- **Left-to-Right Numbering**: Faces are numbered from left to right in the image
- **Visual Results**: Processed images show bounding boxes and labels
- **Responsive Web Interface**: Easy-to-use upload and results pages

## Technical Details

- Uses InsightFace's FaceAnalysis for face detection and embedding generation
- Implements CPU-based inference for broad compatibility
- Supports caching of known face embeddings
- Processes images using OpenCV for visualization
- Built with Flask for the web framework

## File Structure
