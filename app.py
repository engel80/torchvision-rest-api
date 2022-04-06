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
        
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=False)
    app.run(host='0.0.0.0', port=9000, debug=True)
    # app.run()
