import re
import torch
import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.model import Model

model = torch.load('model/mnist.pth')
net = Model()
net.load_state_dict(model)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    global net
    parseImage(request.get_data())
    '''预测'''
    data_transform = transforms.Compose([transforms.ToTensor(), ])
    root = 'static/output.png'
    img = Image.open(root)
    img = img.resize((28,28))
    img = img.convert('L')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # 输入要与model对应
    predict_y = net(img.float()).detach()
    predict_ys = np.argmax(predict_y, axis=-1)
    ans = predict_ys.item()
    print(predict_y)
    print(predict_y.numpy().squeeze()[ans])
    return jsonify(ans)

def get_visit_info(code=0):
    response = {}
    response['code'] = code
    return response

def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./static/output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))

if __name__ == '__main__':
    app.run(debug=True)