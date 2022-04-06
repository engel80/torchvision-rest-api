import io
import json
import torch
import ssl

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# model.to(device)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    # outputs = model.forward(tensor).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

@app.route("/ping")
def ping():
    return 'OK'

@app.route("/cuda")
def cuda():
    if torch.cuda.is_available():
        return "'torch.cuda.is_available': {}, 'torch.cuda.device_coun': {}, 'torch.cuda.current_device': {}, 'torch.cuda.get_device_name': {}".format(str(torch.cuda.is_available()), str(torch.cuda.device_count()), str(torch.cuda.current_device()), torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        return "'torch.cuda.is_available': {}".format(str(False))

import torch
import math

@app.route("/gputest")
def gputest():
    dtype = torch.float
    # device = torch.device("cpu")
    # # device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요

    # 무작위로 입력과 출력 데이터를 생성합니다
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # 무작위로 가중치를 초기화합니다
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        # 순전파 단계: 예측값 y를 계산합니다
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # 손실(loss)을 계산하고 출력합니다
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        # 손실에 따른 a, b, c, d의 변화도(gradient)를 계산하고 역전파합니다.
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # 가중치를 갱신합니다.
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    return f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3'

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=False)
    app.run(host='0.0.0.0', port=9000, debug=True)
    # app.run()
