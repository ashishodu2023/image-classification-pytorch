import logging
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class Trainer:
    def __init__(self, train_loader, val_loader, label_binarizer, model_path, epochs, plot_path,label_bin_path):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_binarizer = label_binarizer
        self.model_path = model_path
        self.epochs = epochs
        self.plot_path = plot_path
        self.label_bin_path = label_bin_path  # Add this attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        logging.info(f"Using device: {self.device}")

    def build_model(self, num_classes):
        """
        Build and prepare the model for training, utilizing GPU if available.
        """
        logging.info("=====================Building the model========================")
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # Freeze base layers

        # Replace the final fully connected layer for our dataset
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        return model.to(self.device)  # Move model to GPU/CPU

    def train(self, model):
        """
        Train the model using the specified training and validation data loaders.
        """
        logging.info("===================Starting training=====================")
        optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9)  # Optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        # For storing training/validation metrics
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch + 1}/{self.epochs}")
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Choose appropriate data loader
                data_loader = self.train_loader if phase == 'train' else self.val_loader
                for inputs, labels in data_loader:
                    # Move data to GPU/CPU
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    optimizer.zero_grad()  # Zero the parameter gradients

                    with torch.set_grad_enabled(phase == 'train'):  # Enable grad only during training
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':  # Backpropagation and optimization
                            loss.backward()
                            optimizer.step()

                    # Update metrics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # Calculate epoch metrics
                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects.double() / len(data_loader.dataset)

                logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Save metrics for plotting
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())

        logging.info("====================Training complete!=========================")
        return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

    def save_results(self, model, train_loss, val_loss, train_acc, val_acc):
        """
        Save the trained model, label binarizer, and training metrics.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        logging.info("=====================Saving model and label binarizer===================")
        # Save the model
        torch.save(model.state_dict(), self.model_path)
        # Save the LabelBinarizer
        with open(self.label_bin_path, "wb") as f:
            pickle.dump(self.label_binarizer, f)


        logging.info("Saving training plot...")
        # Plot training/validation loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(train_loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.plot(train_acc, label="train_acc")
        plt.plot(val_acc, label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.plot_path)
