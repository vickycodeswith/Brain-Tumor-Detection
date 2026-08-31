import os
from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, render_template
from torchvision import transforms, models


# ============================================================
# Flask configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "photos")

app = Flask(
    __name__,
    template_folder="templates"
)

app.secret_key = "brain-tumor-detection"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# Allowed files
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# Classes
# IMPORTANT: Must match training label mapping
# ============================================================

LABELS = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")

else:
    DEVICE = torch.device("cpu")


print("Using device:", DEVICE)


# ============================================================
# Model
# Exact architecture used during training
#
# ResNet50
# 2048 → 512 → 3
# ============================================================

print("Loading ResNet50...")

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 3)
)


# ============================================================
# Load trained weights
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bt_resnet50_model.pt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Image preprocessing
# Must match evaluation preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Prediction
# ============================================================

def get_prediction(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    class_id = predicted.item()

    class_name = LABELS[class_id]

    confidence_percent = (
        confidence.item() * 100
    )

    return (
        class_id,
        class_name,
        confidence_percent
    )


# ============================================================
# Home
# ============================================================

@app.route("/", methods=["GET"])
def main():

    return render_template(
        "DiseaseDet.html"
    )


# ============================================================
# Upload / Prediction
# ============================================================

@app.route(
    "/uimg",
    methods=["GET", "POST"]
)
def uimg():

    if request.method == "GET":

        return render_template(
            "uimg.html"
        )

    if "file" not in request.files:

        return render_template(
            "error.html"
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "error.html"
        )

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "error.html"
        )

    img_bytes = file.read()

    try:

        class_id, class_name, confidence = (
            get_prediction(
                img_bytes
            )
        )

        return render_template(
            "pred.html",
            result=class_name,
            confidence=f"{confidence:.2f}%",
            file=file
        )

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return render_template(
            "error.html"
        ), 500


# ============================================================
# Error handler
# ============================================================

@app.errorhandler(500)
def server_error(error):

    return render_template(
        "error.html"
    ), 500


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )