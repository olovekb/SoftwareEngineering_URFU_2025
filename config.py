import os


class Config:
    PORT = int(os.getenv('APP_PORT', 8080))
    HOST = os.getenv('APP_HOST', '0.0.0.0')
    SECRET_KEY = os.getenv('SECRET_KEY', 'some_random_secret_key')
    UPLOAD_FOLDER = 'uploads'
    STATIC_FOLDER = 'static'

    ALLOWED_EXTENSIONS = {'asc', 'tif', 'tiff'}
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.pt')
    PATCH_SIZE = 200
    SLRM_RAD_CELL = 50
    VE_FACTOR = 5
    MAX_SLOPE_PERCENT = 1.55
    PROXIMITY_M = 5.2
    QUANTILE_PCT = 8.33
    SKEW_LOW = -1.278
    SKEW_HIGH = 1.428
    VARIANCE_THRESH = 0.505
    CIRCULARITY_LEVEL = 0.105
    CIRCULARITY_THRESH = 0.365
    CURVATURE_THRESH = -0.040
    TPI_WINDOW_MULTIPLIER = 11
    TPI_PERCENTILE = 14
    LBP_P = 14
    LBP_R = 2.5
    LBP_ENTROPY_PERCENTILE = 84
