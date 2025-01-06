from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import pickle
import io

app = Flask(__name__)

# Load the model
MODEL_PATH = "model/activity.pth"
LABEL_BINARIZER_PATH = "model/lb.pickle"

# Load the label binarizer
with open(LABEL_BINARIZER_PATH, "rb") as f:
    label_binarizer = pickle.load(f)

# Define the model architecture
def build_model(num_classes):
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Initialize model and load weights
model = build_model(len(label_binarizer.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # Read image and preprocess
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Perform inference
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()

    # Get the predicted label
    label = label_binarizer.classes_[preds.item()]
    response = {
        "label": label,
        "probabilities": probabilities.tolist()
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
