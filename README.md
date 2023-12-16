# API for ML

**Deploy use** : Cloud Run Google CLoud Platform

## TEST THE API

**URL** : `Wait for deploy in Cloud Run`

**Method** : `GET`

**Auth required** : NO

### Success Response

**Condition** : OK.

**Code** : `200`

**Content example**

```
SERVERRRRRRRR.......IS FIREE!!!

```

## PREDICT THE IMAGE

**URL** : ``

**Method** : `POST`

**Auth required** : NO

**Data must provided**

Provide image to be predict with key "image".

**Data example**
All fields must be sent.

```
image = request.files.get("image")

filename = secure_filename(image.filename)
image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
image.save(image_path)

result = predict(image_path)
```

### Success Response

**Condition** : OK.

**Code** : `200`

**Content example**

```json
{
    "data": {
        "confidence_score": 0.9760169386863708,
        "type_prediction": "Plastic",
        "waste_type": "Inorganic"
    },
    "status": {
        "code": 200,
        "message": "Success predicting"
    }
}
```
