from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import shap
from lime import lime_tabular
from typing import Callable, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Télécharger CIFAR-10
train_dataset_cifar = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=preprocess)
test_dataset_cifar = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=preprocess)

# DataLoaders
train_loader_cifar = DataLoader(
    train_dataset_cifar, batch_size=64, shuffle=True)
test_loader_cifar = DataLoader(
    test_dataset_cifar, batch_size=64, shuffle=False)

# Télécharger Fashion MNIST
train_dataset_minst = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=preprocess)
test_dataset_minst = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=preprocess)

# DataLoaders
train_loader_minst = DataLoader(
    train_dataset_minst, batch_size=64, shuffle=True)
test_loader_minst = DataLoader(
    test_dataset_minst, batch_size=64, shuffle=False)


def train_model(nb_epoch, model, data_loader, max_batches=None):
    model.train()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(nb_epoch):
        running_loss = 0.0

        for i, (images, labels) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            print(f"epoch {epoch} - batch {i}: loss = {running_loss}")

        print(
            f"Époque {epoch+1}, perte moyenne : {running_loss / (i + 1):.4f}")


def set_resnet_model(nb_class):
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, nb_class)

    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def split_data(K, dataset):
    # Initialize KFold
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)

    train_loaders = []
    val_loaders = []

    # Split the dataset into 5 folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold}:")
        print(f"Train indices: {train_idx[:5]}")
        print(f"Validation indices: {val_idx[:5]}")

        # Create subsets for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Stocker dans les listes
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        print(
            f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")

    return train_loaders, val_loaders


def predict(model, input):
    with torch.no_grad():
        outputs = model(input)
        _, predicted = torch.max(outputs, 1)

    return predicted


def extract_datapoint_from_loader(data_loader, index=0):
    """
    Extrait un data point spécifique depuis un DataLoader

    Args:
        data_loader: DataLoader contenant vos données
        index: Index du data point à extraire (dans le premier batch)

    Returns:
        x_test: Image extraite (numpy array)
        y_test: Label correspondant
    """
    # Récupérer le premier batch
    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx == 0:  # Premier batch seulement
            # Extraire l'image à l'index spécifié
            x_test = images[index].numpy()  # Convertir en numpy
            y_test = labels[index].item()   # Label correspondant

            print(f"Shape de x_test: {x_test.shape}")
            print(f"Label y_test: {y_test}")

            return x_test, y_test

    return None, None


def prepare_data_for_shap(train_loader, num_background=50, num_test=1):
    """
    Prépare les données au bon format pour SHAP

    Args:
        train_loader: DataLoader d'entraînement
        num_background: Nombre d'échantillons de fond pour SHAP
        num_test: Nombre d'échantillons de test

    Returns:
        x_test: Échantillon(s) de test
        background_data: Données de fond pour SHAP
    """
    all_images = []
    all_labels = []

    # Collecter les données
    for images, labels in train_loader:
        all_images.append(images)
        all_labels.append(labels)

        # Arrêter si on a assez de données
        if len(torch.cat(all_images, dim=0)) >= num_background + num_test:
            break

    # Concaténer
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Séparer background et test
    background_data = all_images[:num_background].numpy()
    x_test = all_images[num_background:num_background + num_test].numpy()

    # Si un seul échantillon de test, enlever la dimension batch
    if num_test == 1:
        x_test = x_test[0]

    print(f"Shape background_data: {background_data.shape}")
    print(f"Shape x_test: {x_test.shape}")

    return background_data


def shap_visualisation(image, model, explainer):

    if len(image.shape) == 3:
        image_batch = image.unsqueeze(0)  # (1, 3, H, W)
    elif len(image.shape) == 4:
        image_batch = image
    else:
        raise ValueError(
            "L'image doit avoir 3 ou 4 dimensions (C,H,W ou B,C,H,W)")

    # Obtenir la prédiction
    with torch.no_grad():
        output = model(image_batch)
        predicted_class = output.argmax(dim=1).item()

    # Obtenir les valeurs SHAP (ancienne API)
    shap_values = explainer.shap_values(image_batch)

    # Extraire et reformater les valeurs SHAP pour la classe prédite
    shap_value = shap_values[predicted_class][0]  # (3, H, W)
    shap_value_hwc = np.transpose(shap_value, (1, 2, 0))  # (H, W, C)

    # Image originale en format HWC
    image_np = image_batch[0].cpu().numpy().transpose(1, 2, 0)

    return shap_value_hwc, image_np
