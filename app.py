import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1.pt", map_location=torch.device("cpu")))

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

@app.route('/')
def home_page():
    return "working"

@app.route('/api/start')
def start():
    return "start"

@app.route('/api/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # image = request.files['image']
        filename = "apple_black_rot.JPG"
        file_path = os.path.join('static/uploads', filename)
        # image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
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
    app.run(debug=True, port=8000)
