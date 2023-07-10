from flask import Flask, request
from test import test
import numpy as np
import cv2
import base64

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False) 