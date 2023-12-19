import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from swagger import create_swagger_blueprint
from flask_cors import CORS
from ultralytics import YOLO

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "http://127.0.0.1:8080"}})

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'static/uploads/WasteWise'
app.config['MODEL_FILE'] = 'model/model_WasteWise.h5'
app.config['PLASTIC_MODEL_FILE'] = 'model/model_plasticType.pt'

SWAGGER_URL = '/api_model-docs'
swagger_ui_blueprint = create_swagger_blueprint()

# Load Model for Prediction WasteWise Model
wastewise_model_file_path = app.config['MODEL_FILE']
if not os.path.exists(wastewise_model_file_path):
    print(f"Error: Model file '{wastewise_model_file_path}' not found.")

wastewise_model = tf.keras.models.load_model(wastewise_model_file_path, custom_objects={"KerasLayer": tfhub.KerasLayer})

labels = {
    0: "Biological",
    1: "Cardboard",
    2: "Glass",
    3: "Metal",
    4: "Paper",
    5: "Plastic"
}

# Load Model for Plastic Type Prediction
plastic_model_file_path = app.config['PLASTIC_MODEL_FILE']
if not os.path.exists(plastic_model_file_path):
    print(f"Error: Model file '{plastic_model_file_path}' not found.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_waste_type(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img)

        normalized_image_array = img_array / 255.0

        # Expand dimensions to match the input shape of the model
        input_data = np.expand_dims(normalized_image_array, axis=0).astype(np.float32)

        # Make predictions using the loaded model
        predictions = wastewise_model.predict(input_data)

        # Determine the predicted class and confidence score
        class_index = np.argmax(predictions)
        predicted_class = labels[class_index]
        confidence_score = float(predictions[0, class_index])

        plastic_type = None
        if(predicted_class == "Plastic"):
            plastic_model = YOLO(plastic_model_file_path)

            plastic_prediction = plastic_model.predict(img_path, imgsz=160)

            # Get the prediction with the highest confidence score
            top_prediction = plastic_prediction[0].probs.top1

            # Map the class index to the plastic type
            mapping = {0: 'HDPE', 1: 'LDPE', 2: 'PET', 3: "PP", 4: "PS", 5: "PVC"}
            plastic_type = mapping[top_prediction]
        
        # Map waste_id based on the predicted class
        waste_id_mapping = {
            "Biological": "biological",
            "Cardboard": "cardboard",
            "Glass": "glass",
            "Metal": "metal",
            "Paper": "paper",
            "Plastic": f"plastic_{plastic_type.lower()}" if plastic_type else "plastic"
        }
        
        waste_id = waste_id_mapping.get(predicted_class, "unknown")

        result = {"type_prediction": predicted_class, "confidence_score": confidence_score, "plastic_type": plastic_type, "waste_id": waste_id, "error": None}
        return result

    except Exception as e:
        return {
            "type_prediction": None,
            "confidence_score": None,
            "error": f"Error processing image: {str(e)}",
        } 

@app.route("/")
def index():
    return "SERVERRRRRRRR.......IS FIREE!!!"

@app.route('/prediction', methods=['POST'])
def prediction_route():
    if request.method == "POST":

        # image = request.files["image"]
        image = request.files.get("image")

        if image is None or image.filename == '':
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "No file provided in the request."
                }
            }), 400

        if not allowed_file(image.filename):
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
                }
            }), 400

        try:
            # Save the uploaded image
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            # Call the function to predict waste type
            result = predict_waste_type(image_path)
            
            if result["type_prediction"] == "Plastic":
                return jsonify({
                    "status": {
                        "code": 200,
                        "message": "Success predicting"
                    },
                    # "data": {
                    #     "confidence_score": result["confidence_score"],
                    #     "waste_id": result["waste_id"]
                    # }
                    "data": {
                        "type_prediction": result["type_prediction"],
                        "plastic_type": result["plastic_type"],
                        "confidence_score": result["confidence_score"],
                        "waste_id": result["waste_id"]
                    }
                }), 200
            else:
                return jsonify({
                    "status": {
                        "code": 200,
                        "message": "Success predicting"
                    },
                    # "data": {
                    #     "confidence_score": result["confidence_score"],
                    #     "waste_id": result["waste_id"]
                    # }
                    "data": {
                        "type_prediction": result["type_prediction"],
                        "confidence_score": result["confidence_score"],
                        "waste_id": result["waste_id"]
                    }
                }), 200

        except Exception as e:
            return jsonify({
                "status": {
                    "code": 500,
                    "message": f"Internal server error: {str(e)}"
                }
            }), 500

    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            }
        }), 405

if __name__ == "__main__":
    app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))