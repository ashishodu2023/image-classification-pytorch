import logging
import cv2
import torch
import numpy as np
import pickle
from torchvision import models, transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class Predictor:
    def __init__(self, model_path, label_bin_path, num_classes):
        self.model_path = model_path
        self.label_bin_path = label_bin_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

    def load_model(self):
        logging.info("=================Loading model and label binarizer=====================")
        
        # Define the model architecture (e.g., ResNet50)
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        #model.fc = torch.nn.Linear(num_features, self.num_classes)
        # Recreate the custom classification head
        model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),  # fc.0
        torch.nn.ReLU(),                    # fc.1
        torch.nn.Dropout(0.5),              # fc.2
        torch.nn.Linear(512, self.num_classes)   # fc.3
    )
        # Load the model weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Load the label binarizer
        with open(self.label_bin_path, "rb") as f:
            label_binarizer = pickle.load(f)
        
        return model, label_binarizer

    def classify_image(self, image_path):
        model, lb = self.load_model()
        model.eval()
        model.to(self.device)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = transform(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            preds = model(image_transformed)
            probs = torch.nn.functional.softmax(preds, dim=1)

        # Get the predicted label
        label_idx = torch.argmax(probs, dim=1).item()
        label = lb.classes_[label_idx]

        # Log and return results
        logging.info(f"Predicted label: {label}")
        return label, probs.cpu().numpy()
