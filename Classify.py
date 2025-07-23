import os
import torch
import torch.nn as nn
from torchvision.models import convnext_base
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATHS = ["Classifier.pt"]  # Add more model paths here
IMAGE_DIR = "test_images"
IMAGE_SIZE = (360, 360)

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# === LOAD MODELS ===
def load_model(model_path):
    model = convnext_base(pretrained=False)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

models = [load_model(path) for path in MODEL_PATHS]

# === LABELING ===
def get_prediction_label(output):
    return "Low-light" if output >= 0.5 else "Well-lit"

# === INFERENCE FUNCTION ===
def classify_and_show(image_paths):
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        predictions = []
        for idx, model in enumerate(models, 1):
            with torch.no_grad():
                output = torch.sigmoid(model(input_tensor)).item()
                label = get_prediction_label(output)
                predictions.append(f"(Confidence: {output:.4f})")

        # === SHOW IMAGE + ALL PREDICTIONS ===
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title("\n".join(predictions), fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# === RUN ===
if __name__ == "__main__":
    image_paths = [
        os.path.join(IMAGE_DIR, fname)
        for fname in os.listdir(IMAGE_DIR)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    classify_and_show(image_paths)
