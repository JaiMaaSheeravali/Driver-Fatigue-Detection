import io
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import numpy as np
import cv2


def initialize_model(use_pretrained=True):
    model = torchvision.models.resnet50(pretrained=use_pretrained)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.Linear(256, 4)
    )
    return model


app = Flask(__name__)

drowsiness_class_index = json.load(open('./drowsiness_class_index.json'))

PATH = './models/resnet50_model.pth'
model_cnn = initialize_model(use_pretrained=False)
model_cnn.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model_cnn.eval()


cvmodelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
cvconfigFile = "models/deploy.prototxt.txt"
cvnet = cv2.dnn.readNetFromCaffe(cvconfigFile, cvmodelFile)


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.4722, 0.4101, 0.3794],
                                            [0.2401, 0.2342, 0.2323])])
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return my_transforms(image).unsqueeze(0)


def get_prediction(image):
    # apply transformations to input image
    tensor = transform_image(image=image)

    # send the image to resnet model
    outputs = model_cnn.forward(tensor)

    # calculate probabilities of each class
    outputs = F.softmax(outputs, dim=1)

    # get the max probability class
    y_hat, y_idx = outputs.max(1)
    predicted_prob = str(y_hat.item())
    predicted_idx = str(y_idx.item())

    return predicted_prob, drowsiness_class_index[predicted_idx]


def extract_face(img_bytes):
    # convert img_bytes to numpy array for opencv
    nparr = np.frombuffer(img_bytes, dtype=np.uint8)
    # open nparryay using opencv
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (123.68, 116.77, 103.93))

    # pass the blob through the network and obtain the detections and predictions
    cvnet.setInput(blob)
    detections = cvnet.forward()

    frames = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.7:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            frame = None
            if startY > h and startX > w:
                frame = image
            elif startY > h:
                frame = image[:, startX: endX]
            elif startX > w:
                frame = image[startY: endY, :]
            else:
                frame = image[startY:endY, startX:endX]
            frames.append(frame)

    if(len(frames) > 0):
        return frames[0]
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']

        # convert that to bytes
        img_bytes = file.read()

        face = extract_face(img_bytes)
        cv2.imwrite(file.filename, face)

        probability, class_name = get_prediction(image=face)
        return jsonify({'probability': probability, 'class_name': class_name})
