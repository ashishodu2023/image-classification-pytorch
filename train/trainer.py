import logging
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import pickle
from torch.cuda.amp import GradScaler, autocast
import mlflow


class Trainer:
    def __init__(self, train_loader, val_loader, label_binarizer, model_path, epochs, plot_path, label_bin_path):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_binarizer = label_binarizer
        self.model_path = model_path
        self.epochs = epochs
        self.plot_path = plot_path
        self.label_bin_path = label_bin_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()  # For mixed precision training
        logging.info(f"Using device: {self.device}")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("image-classification")

    def build_model(self, num_classes):
        logging.info("=====================Building the model========================")
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        print(model)
        return model.to(self.device)

    def train(self, model):
        logging.info("===================Starting training=====================")
        optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        best_loss = float("inf")
        patience = 5
        patience_counter = 0

        with mlflow.start_run():
            #mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])

            for epoch in range(self.epochs):
                logging.info(f"Epoch {epoch + 1}/{self.epochs}")
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    running_corrects = 0
                    data_loader = self.train_loader if phase == 'train' else self.val_loader

                    for inputs, labels in data_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        optimizer.zero_grad()
                        with torch.amp.autocast(device_type='cuda'):  # Mixed precision training
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        if phase == 'train':
                            self.scaler.scale(loss).backward()
                            self.scaler.step(optimizer)
                            self.scaler.update()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(data_loader.dataset)
                    epoch_acc = running_corrects.double() / len(data_loader.dataset)

                    logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                    mlflow.log_metric(f"{phase}_accuracy", epoch_acc.item(), step=epoch)

                    if phase == 'train':
                        train_loss_history.append(epoch_loss)
                        train_acc_history.append(epoch_acc.item())
                        scheduler.step()
                    else:
                        val_loss_history.append(epoch_loss)
                        val_acc_history.append(epoch_acc.item())

                        if epoch_loss < best_loss:
                            best_loss = epoch_loss
                            patience_counter = 0
                            torch.save(model.state_dict(), self.model_path)
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            logging.info("Early stopping triggered.")
                            break

            mlflow.pytorch.log_model(model, "model")

        logging.info("====================Training complete!=========================")
        return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

    def save_results(self, model, train_loss, val_loss, train_acc, val_acc):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        logging.info("=====================Saving model and label binarizer===================")
        torch.save(model.state_dict(), self.model_path)
        with open(self.label_bin_path, "wb") as f:
            pickle.dump(self.label_binarizer, f)

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
        #mlflow.log_artifact(self.plot_path)
        # Log artifacts to MLflow
        with mlflow.start_run():
            logging.info("================Logging model and artifacts to MLflow======================")
            # Log the trained model to MLflow
            mlflow.pytorch.log_model(model, artifact_path="model")
            # Log the training plot
            mlflow.log_artifact(self.plot_path)
            # Log the label binarizer
            mlflow.log_artifact(self.label_bin_path)
            logging.info("=================Model and artifacts logged successfully===================")
