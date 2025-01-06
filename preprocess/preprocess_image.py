from PIL import Image
import os

def clean_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Convert to RGB to standardize format
            img = img.convert("RGB")
            # Re-save the image, stripping metadata
            img.save(file_path, format="JPEG", optimize=True)
            print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

dataset_path = "data/"  # Adjust to your dataset path

for root, _, files in os.walk(dataset_path):
    for file in files:
        # Process files with jpg, jpeg, or png extensions
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(root, file)
            clean_image(file_path)
