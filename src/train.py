import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 25
WARMUP_EPOCHS = 2

BACKBONE_LR = 0.001
HEAD_LR = 0.01

NUM_CLASSES = 3
IMAGE_SIZE = 224

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset
# ============================================================

class BrainTumorDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        # Convert dataset labels:
        # 1 -> 0 Meningioma
        # 2 -> 1 Glioma
        # 3 -> 2 Pituitary
        self.label_map = {
            1: 0,
            2: 1,
            3: 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]
        original_label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        label = self.label_map[original_label]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load split files
# ============================================================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

print("\nDataset:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


train_dataset = BrainTumorDataset(
    train_df,
    transform=train_transform
)

val_dataset = BrainTumorDataset(
    val_df,
    transform=eval_transform
)

test_dataset = BrainTumorDataset(
    test_df,
    transform=eval_transform
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# ResNet50
# ============================================================

print("\nLoading ResNet50...")

try:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
except AttributeError:
    model = models.resnet50(pretrained=True)


# Replace classifier
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, NUM_CLASSES)
)

model = model.to(DEVICE)

print("ResNet50 loaded successfully.")
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

# Warm-up:
# Freeze backbone and train classifier only.

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.fc.parameters():
    parameter.requires_grad = True


optimizer = torch.optim.Adam(
    [
        {
            "params": model.fc.parameters(),
            "lr": HEAD_LR
        }
    ]
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# Checkpoint directory
# ============================================================

os.makedirs("models", exist_ok=True)

best_val_accuracy = 0.0


# ============================================================
# Training
# ============================================================

print("\nStarting training...")
print("Warm-up epochs:", WARMUP_EPOCHS)
print("Total epochs:", EPOCHS)

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Unfreeze backbone after warm-up
    # --------------------------------------------------------

    if epoch == WARMUP_EPOCHS:

        print("\nUnfreezing ResNet50 backbone...")

        for parameter in model.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.fc.parameters(),
                    "lr": HEAD_LR
                },
                {
                    "params": [
                        p for name, p in model.named_parameters()
                        if not name.startswith("fc.")
                    ],
                    "lr": BACKBONE_LR
                }
            ]
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=7,
            gamma=0.1
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    # Scheduler only after warm-up
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/bt_resnet50_model.pt"
        )

        print(
            f"✓ Best model saved "
            f"(Validation Accuracy: {val_accuracy:.4f})"
        )


# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        "models/bt_resnet50_model.pt",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# Final Test Evaluation
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

test_accuracy = (
    np.array(all_predictions) ==
    np.array(all_labels)
).mean()

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nBest Validation Accuracy:", best_val_accuracy)
print("Model saved to: models/bt_resnet50_model.pt")