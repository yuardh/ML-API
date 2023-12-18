from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads/WasteWise'

# Set up the Ultralytics YOLO model
model = YOLO("model/model_plasticType.pt")

# Mapping for plastic types
mapping = {0: 'HDPE', 1: 'LDPE', 2: 'PET', 3: 'PP', 4: 'PS', 5: 'PVC'}

@app.route('/')
def index():
    return "SERVERRRRRRRR PLASTIC_TYPE.......IS FIREE!!!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400

        image = request.files['image']

        # Check if the file is empty
        if image.filename == '':
            return jsonify({"error": "No file provided in the request."}), 400

        try:
            # Save the uploaded image
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            # Perform YOLO prediction
            results = model.predict(image_path, imgsz=160)

            # Get the top prediction
            top_prediction = results[0].probs.top1

            # Get the plastic type from the mapping
            plastic_type = mapping[top_prediction]

            return jsonify({"success": True, "plastic_type": plastic_type})

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
