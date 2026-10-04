"""
Object Detection App - Flask + OpenCV + EasyOCR
Deployed to Fly.io
"""
import os
import sqlite3
import uuid
from datetime import datetime

import cv2
import easyocr
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
DATABASE = os.environ.get("DATABASE", "objectdetection.db")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
init_db()

# Initialize EasyOCR reader (English)
reader = easyocr.Reader(["en"], gpu=False)


def get_db():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with detections table."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            objects_detected TEXT NOT NULL,
            text_detected TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_objects(image_path):
    """Detect objects using OpenCV DNN with MobileNet SSD."""
    # Load pre-trained MobileNet SSD model
    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"

    # Check if model files exist, if not use simple contour detection
    if not os.path.exists(prototxt) or not os.path.exists(model):
        return detect_objects_simple(image_path)

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Class names for MobileNet SSD
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    img = cv2.imread(image_path)
    if img is None:
        return [], 0.0

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    objects = []
    max_confidence = 0.0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_name = classes[class_id] if class_id < len(classes) else "unknown"
            objects.append({
                "class": class_name,
                "confidence": round(float(confidence), 3)
            })
            max_confidence = max(max_confidence, confidence)

    return objects, round(max_confidence, 3)


def detect_objects_simple(image_path):
    """Simple object detection using contour detection as fallback."""
    img = cv2.imread(image_path)
    if img is None:
        return [], 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            objects.append({
                "class": f"object_{i+1}",
                "confidence": round(min(area / 10000, 0.99), 3),
                "bbox": [x, y, w, h]
            })

    confidence = max((obj["confidence"] for obj in objects), default=0.0)
    return objects, confidence


def detect_text(image_path):
    """Detect text using EasyOCR."""
    try:
        results = reader.readtext(image_path)
        texts = []
        for (bbox, text, conf) in results:
            texts.append({
                "text": text,
                "confidence": round(float(conf), 3)
            })
        return texts
    except Exception as e:
        import logging
        logging.error(f"EasyOCR error: {e}")
        return []


@app.route("/")
def index():
    """Render main page."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Handle image upload and object detection."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Save uploaded file
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run detection
    objects, confidence = detect_objects(filepath)
    texts = detect_text(filepath)

    # Save to database
    conn = get_db()
    conn.execute(
        "INSERT INTO detections (filename, original_name, objects_detected, text_detected, confidence) VALUES (?, ?, ?, ?, ?)",
        (filename, file.filename, str(objects), str(texts), confidence)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "filename": filename,
        "original_name": file.filename,
        "objects_detected": objects,
        "text_detected": texts,
        "confidence": confidence
    })


@app.route("/history")
def history():
    """Get detection history."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "filename": row["filename"],
            "original_name": row["original_name"],
            "objects_detected": row["objects_detected"],
            "text_detected": row["text_detected"],
            "confidence": row["confidence"],
            "created_at": row["created_at"]
        })

    return jsonify(history)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
