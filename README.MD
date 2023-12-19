# bangkit-machine-learning

The source code of machine learning model's API of WasteWise in order to complete Bangkit Capstone Project

# API URL
[Flask API]

# API Endpoint
|     Endpoint    |   Method   |  Body Sent (JSON)  |              Description              |
|     :------     | :--------: |  :--------------:  | :-----------------------------------: |
|        /        |    GET     |        None        |   HTTP GET REQUEST Testing Endpoint   |
|   /prediction   |    POST    |  image: Image file |          Prediction Endpoint          |
| /api_model-docs |    GET     |    swagger.json    |           API Documentations          |

# How to run this Flask app
- Clone this repo
- Open terminal and go to this project's root directory
- Type `python -m venv .venv` and hit enter
- In Linux, type `source .venv/bin/activate`
- In Windows, type `.venv\Scripts\activate`
- Type `pip install -r requirements.txt`
- Serve the Flask app by typing `flask run` or `python app.py` for debungging mode
- When typing `flask run` it will run on `http://127.0.0.1:5000`
- When typing `python app.py` it will run on `http://127.0.0.1:8080`

# How to predict image with Postman
- Open Postman
- Enter URL request bar with `http://127.0.0.1:8080/prediction`
- Select method POST
- Go to Body tab and select form-data
- Change key from form-data with file (it must be named `image`)
- Input the image that you want predict as a value of the key
- Send the request

**Success response - Non Plastic**

```json
{
    "data": {
        "confidence_score": 0.9924584031105042,
        "type_prediction": "Cardboard",
        "waste_id": "cardboard"
    },
    "status": {
        "code": 200,
        "message": "Success predicting"
    }
}

**Success response - Plastic**

```json
{
    "data": {
        "confidence_score": 0.9924584031105042,
        "plastic_type": "PET",
        "type_prediction": "Plastic",
        "waste_id": "plastic_pet"
    },
    "status": {
        "code": 200,
        "message": "Success predicting"
    }
}
