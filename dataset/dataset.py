import cv2
import torch
from torch.utils.data import Dataset
import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class SportsDataset(Dataset):
    """
    Custom Dataset class to load images for training and validation.
    """

    def __init__(self, image_paths, labels, transform=None, log_samples=0):
        """
        Initialize the dataset with image paths and corresponding labels.
        :param image_paths: List of paths to image files.
        :param labels: List of labels corresponding to the images.
        :param transform: Transformations to apply to the images.
        :param log_samples: Number of initial samples to log for debugging.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.log_samples = log_samples  # Number of samples to log (for debugging)


    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetch a single sample (image and label) by index.
        :param idx: Index of the sample to fetch.
        :return: Transformed image and corresponding label.
        """
        # Load the image
        image_path = self.image_paths[idx]
        if idx < self.log_samples:
            print(f"[DEBUG] Loading image from: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded. Check the file or path.")

        # Ensure the image is in RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get the corresponding label
        label = self.labels[idx]

        return image, label
