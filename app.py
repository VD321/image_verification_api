from flask import Flask, request
from test import test
import numpy as np
import cv2
import base64
import multiprocessing
from gunicorn.app.base import BaseApplication
import os

import torch

from src.utility import get_kernel, parse_model_name
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE

app = Flask(__name__)

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    image_base64 = data['file']
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    result = test(image, "./resources/anti_spoof_models", models)
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
        initialize_worker() 
        return self.application
    
def run_gunicorn(app):
    options = {
        'bind': '0.0.0.0:5000', 
        'workers': get_num_of_workers(),
        'worker_class': 'gthread'
    }

    GunicornApp(app, options).run()

models = {}

def load_model(model_dir):
    global models
    device = torch.device("cpu")

    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input,)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

        models[model_path] = model

        # load model weight
        state_dict = torch.load(model_path, map_location=device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
            model.eval()
        else:
            model.load_state_dict(state_dict)

def load_caffemodel ():
    caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
    deploy = "./resources/detection_model/deploy.prototxt.txt"
    detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
    models[caffemodel] = detector

def initialize_worker():
    load_model("./resources/anti_spoof_models")
    load_caffemodel()

if __name__ == "__main__":
    run_gunicorn(app)