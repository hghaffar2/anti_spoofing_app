import os
from flask import request, Flask,jsonify, redirect, url_for, render_template
import requests
import json
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import urllib.request

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

cdir = os.path.join(os.path.dirname(__file__))
model_dir = cdir + "/resources/anti_spoof_models/"

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	data = {'label': "unknown", 'value':0.0}
	if request.method == 'POST':
		url = request.form['url']
		url_response = urllib.request.urlopen(url)
		img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
		img = cv2.imdecode(img_array, -1)

		label,value = test(img)
		if label == 1:
			data['label'] = "real"
		else:
			data['label'] = "fake"
		data['value'] = value

	return jsonify(label=data['label'], value=data['value'])


def load_models():
    model_test = AntiSpoofPredict()
    return model_test
def load_utils():
    image_cropper = CropImage()
    return image_cropper
def test(image):
    # result = check_image(image)
    # if result is False:
    #     return
    image_bbox = model.get_bbox(image)
    prediction = np.zeros((1, 3))
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model.predict(img, os.path.join(model_dir, model_name))

    # result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    return label, value
    
    # if label == 1:
    #     print("Image is Real Face. Score: {:.2f}.".format(value))
    #     result_text = "RealFace Score: {:.2f}".format(value)
    # else:
    #     print("Image is Fake Face. Score: {:.2f}.".format(value))
    #     result_text = "FakeFace Score: {:.2f}".format(value)



@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
	model = load_models()
	image_cropper = load_utils()

	app.run( host="127.0.0.1", port=int("8081"))