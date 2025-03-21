import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchvision.models import vgg16
from tqdm import tqdm

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for VGG16
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize as expected by VGG16
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./../datasets', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)

# Load pretrained VGG16 model
model = vgg16(pretrained=True).to(device)
model.eval()  # Set to evaluation mode

# Remove classifier layers
feature_extractor = nn.Sequential(*list(model.features.children())).to(device)

# Function to extract features
def extract_features(dataloader, feature_extractor, device):
    features = []
    labels = []
    for images, label in tqdm(dataloader, desc="Extracting Features"):
        images = images.to(device)
        with torch.no_grad():
            print("v1")
            deep_features = feature_extractor(images)  # Get deep features
            print("v2")
            deep_features = deep_features.view(images.size(0), -1).cpu().numpy()  # Flatten and move to CPU
            print("v3")
            features.append(deep_features)
            print("v4")
            labels.extend(label.numpy())
            print("v5")
    print(f'features = {features}')
    features = np.vstack(features)  # Convert list to array
    print("v6")
    labels = np.array(labels)
    print("v7")
    return features, labels

# Extract features
try:
    print('extraxting features')
    features, labels = extract_features(dataloader, feature_extractor, device)
    print("v8")
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    print('done')
except Exception as e:
    print(f'Sorry: {e}')
# Evaluate model
preds = knn.predict(features)
accuracy = accuracy_score(labels, preds)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(labels, preds))
