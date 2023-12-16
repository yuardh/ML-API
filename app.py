import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from swagger import create_swagger_blueprint
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": ["http://192.168.1.3:8080", "http://127.0.0.1:8080"]}})

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'static/uploads/WasteWise'
app.config['MODEL_FILE'] = 'model_hana.h5'

SWAGGER_URL = '/api_model-docs'
swagger_ui_blueprint = create_swagger_blueprint()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

model_file_path = app.config['MODEL_FILE']
if not os.path.exists(model_file_path):
    print(f"Error: Model file '{model_file_path}' not found.")


model = tf.keras.models.load_model(model_file_path, custom_objects={"KerasLayer": tfhub.KerasLayer})
tf.saved_model.save(model, 'saved_model')  
loaded_model = tf.saved_model.load('saved_model')  

labels = {
    0: "Biological",
    1: "Cardboard",
    2: "Glass",
    3: "Metal",
    4: "Paper",
    5: "Plastic"
}

def predict(img):
    try:
        img = Image.open(img).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img)

        normalized_image_array = img_array / 255.0

        # Expand dimensions to match the input shape of the model
        input_data = np.expand_dims(normalized_image_array, axis=0).astype(np.float32)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Determine the predicted class and confidence score
        class_index = np.argmax(predictions)
        predicted_class = labels[class_index]
        confidence_score = float(predictions[0, class_index])

        # Determine whether the waste is organic or inorganic
        waste_type = "Organic" if class_index == 0 else "Inorganic"

        result = {"type_prediction": predicted_class, "waste_type": waste_type, "confidence_score": confidence_score, "error": None}

        return result

    except Exception as e:
        return {"type_prediction": None, "waste_type": None, "confidence_score": None, "error": f"Error processing image: {str(e)}"}

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
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            # Call the function to predict waste type
            result = predict(image_path)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "type_prediction": result["type_prediction"],
                    "waste_type": result["waste_type"],
                    "confidence_score": result["confidence_score"]
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