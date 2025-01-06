import argparse
import logging
from train.trainer import Trainer
from predict.predictor import Predictor
from dataset.dataset import SportsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Main:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Image Classification")
        self.parser.add_argument(
            "--task", choices=["train", "predict"], required=True, help="Task to perform: train or predict"
        )
        self.parser.add_argument("--dataset", help="Path to dataset (for training)")
        self.parser.add_argument("--model", help="Path to save/load model")
        self.parser.add_argument("--label-bin", help="Path to save/load label binarizer")
        self.parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (for training)")
        self.parser.add_argument("--plot", help="Path to save training plot")
        self.parser.add_argument("--image", help="Path to input image (for prediction)")
        self.parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
        self.args = self.parser.parse_args()

    def run(self):
        if self.args.task == "train":
            self.train()
        elif self.args.task == "predict":
            self.predict()

    def train(self):
        logging.info("================Preparing dataset for training==================")
        # Load image paths and corresponding labels
        image_paths = list(paths.list_images(self.args.dataset))
        labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

        # Encode labels using LabelBinarizer
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels).argmax(axis=1)

        # Split dataset into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.25, stratify=labels, random_state=42
        )

        # Define data augmentation and normalization transformations
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Create dataset and data loaders
        train_dataset = SportsDataset(train_paths, train_labels, transform=train_transforms)
        val_dataset = SportsDataset(val_paths, val_labels, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        label_bin_path = "model/lb.pickle"

        # Initialize the Trainer and start training
        trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            label_binarizer=lb,
            label_bin_path=label_bin_path,
            model_path=self.args.model,
            epochs=self.args.epochs,
            plot_path=self.args.plot,
        )
        model = trainer.build_model(len(lb.classes_))
        model, train_loss, val_loss, train_acc, val_acc = trainer.train(model)
        trainer.save_results(model, train_loss, val_loss, train_acc, val_acc)

    def predict(self):
        logging.info("====================Starting image prediction======================")
        # Initialize the Predictor and classify the input image
        predictor = Predictor(self.args.model, self.args.label_bin, self.args.num_classes)
        label, probabilities = predictor.classify_image(self.args.image) 
        print(f"Predicted Label: {label}")
        print(f"Probabilities: {probabilities}")



if __name__ == "__main__":
    main = Main()
    main.run()
