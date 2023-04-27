import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import boto3
import io
import numpy as np
import torch
import pandas as pd
import CNN
from smart_open import smart_open
import requests
from tempfile import NamedTemporaryFile


disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

s3 = boto3.client('s3',
                  aws_access_key_id='AKIATFLRHDCN3KRSS7CQ',
                  aws_secret_access_key='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',
                  region_name='ap-south-1')

model = None  # initialize model object


def load_model():
    global model
    load_path = "s3://rootskart-users/plant-disease-model/plant_disease_model_1.pt"
    with smart_open(load_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer, map_location=torch.device('cuda'))
    model = CNN.CNN(39)
    model.load_state_dict(state_dict)
    model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)


@app.before_first_request
def startup():
    load_model()


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello World!'})


@app.route('/api/v1/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_stream = io.BytesIO(image.read())
        s3.upload_fileobj(
            file_stream, 'rootskart-disease-prediction', filename)
        image_url = s3.generate_presigned_url(
            'get_object', Params={'Bucket': 'rootskart-disease-prediction', 'Key': filename})
        # image = Image.open(file_stream)
        # file_path = os.path.join('static/uploads', filename)
        with NamedTemporaryFile(delete=False) as f:
            response = requests.get(image_url)
            f.write(response.content)
        pred = prediction(f.name)
        os.unlink(f.name)
        # image.save(image_url)
        # print(image_url)
        # pred = prediction(image)
        # pred = prediction(image_url)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return jsonify({
            'title': title,
            'description': description,
            'prevent': prevent,
            'pred': int(pred),
            'supplement_name': supplement_name,
            'supplement_image_url': supplement_image_url,
            'supplement_buy_link': supplement_buy_link
        })


if __name__ == '__main__':
    app.run(port=8000)
