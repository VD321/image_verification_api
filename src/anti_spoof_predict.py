import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

def get_bbox(img, models):
    detector = models["./resources/detection_model/Widerface-RetinaFace.caffemodel"]

    height, width = img.shape[0], img.shape[1]
    aspect_ratio = width / height
    if img.shape[1] * img.shape[0] >= 192 * 192:
        img = cv2.resize(img,
                         (int(192 * math.sqrt(aspect_ratio)),
                          int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
    detector.setInput(blob, 'data')
    out = detector.forward('detection_out').squeeze()
    max_conf_index = np.argmax(out[:, 2])
    left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                               out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
    bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
    return bbox


def predict(img, model_path, models):
    device = torch.device("cpu")
    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    img = test_transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        result = models[model_path].forward(img)
        result = F.softmax(result).cpu().numpy()
    return result

