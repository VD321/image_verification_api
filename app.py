from flask import Flask, request
from test import test
import numpy as np
import cv2
import base64
import multiprocessing
from gunicorn.app.base import BaseApplication

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    image_base64 = data['file']
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    result = test(image, "./resources/anti_spoof_models", 0)
    return {"faceType" : result.faceType, "score" : result.score, "predictionCost" : result.predictionCost}

def get_num_of_workers():
    return (multiprocessing.cpu_count() * 2 + 1)

class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application
    
def run_gunicorn(app):
    options = {
        'bind': '0.0.0.0:5000', 
        'workers': get_num_of_workers(),
    }

    GunicornApp(app, options).run()


if __name__ == "__main__":
    run_gunicorn(app)