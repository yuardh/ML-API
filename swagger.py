from flask_swagger_ui import get_swaggerui_blueprint

def create_swagger_blueprint():
    SWAGGER_URL = '/api_model-docs'
    API_URL = '/static/API-Documentations/swagger.json'

    return get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "WasteWise Machine Learning API"
        }
    )
