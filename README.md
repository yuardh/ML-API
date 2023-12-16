# API for ML

**Deploy use** : Cloud Run Google Cloud Platform

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
  "result": "cardboard"
}
```

### Referensi

- [faizan170/tensorflow-image-classification-flask-deployment](https://github.com/faizan170/tensorflow-image-classification-flask-deployment "faizan170's Github profile")
- [how-to-deploy-a-simple-flask-app-on-cloud-run-with-cloud-endpoint](https://medium.com/fullstackai/how-to-deploy-a-simple-flask-app-on-cloud-run-with-cloud-endpoint-e10088170eb7 "simple-flask-app-on-cloud-run")
