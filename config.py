import os


class Config:
    PORT = int(os.getenv('APP_PORT', 8080))
    HOST = os.getenv('APP_HOST', '0.0.0.0')
    SECRET_KEY = 'some_random_secret_key'
    UPLOAD_FOLDER = 'uploads'
    STATIC_FOLDER = 'static'

    ALLOWED_EXTENSIONS = {'asc', 'tif', 'tiff'}
    MODEL_PATH = 'model.pt'
    PATCH_SIZE = 200
    SLRM_RAD_CELL = 50
    VE_FACTOR = 5
    MAX_SLOPE_PERCENT = 1.55
    PROXIMITY_M = 5.2
    QUANTILE_PCT = 8.33
