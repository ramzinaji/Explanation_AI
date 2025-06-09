import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.model_selection import KFold
import numpy as np
import shap
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
import os


# 1️⃣ Classe de prétraitement
class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_dataset(self, name="CIFAR10", train=True):
        if name == "CIFAR10":
            return datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)
        elif name == "FashionMNIST":
            return datasets.FashionMNIST(root='./data', train=train, download=True, transform=self.transform)
        else:
            raise ValueError("Dataset non supporté")

    def create_dataloader(self, dataset, batch_size=64, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def split_data(self, dataset, K=5):
        kfold = KFold(n_splits=K, shuffle=True, random_state=42)
        train_loaders, val_loaders = [], []

        for train_idx, val_idx in kfold.split(dataset):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loaders.append(DataLoader(
                train_subset, batch_size=64, shuffle=True))
            val_loaders.append(DataLoader(
                val_subset, batch_size=64, shuffle=False))

        return train_loaders, val_loaders


# 2️⃣ Classe d'entraînement et modèle
class Trainer:
    def __init__(self, num_classes):
        self.model = self.set_resnet_model(num_classes)

    def set_resnet_model(self, num_classes):
        model = resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def train(self, data_loader, epochs=5, max_batches=None):
        self.model.train()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(data_loader):
                if max_batches is not None and i >= max_batches:
                    break

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch} Batch {i}: Loss = {loss.item():.4f}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            return torch.argmax(output, dim=1).item()


# 3️⃣ Classe pour SHAP
class Explainer:
    def __init__(self, model, background_data):
        self.explainer = shap.GradientExplainer(model, background_data)
        self.model = model

    def prepare_background(self, loader, num_background=50, num_test=1):
        images_accum = []
        for images, _ in loader:
            images_accum.append(images)
            if len(torch.cat(images_accum)) >= num_background + num_test:
                break
        all_images = torch.cat(images_accum, dim=0)
        background = all_images[:num_background]
        test_sample = all_images[num_background:num_background + num_test]
        return background, test_sample

    def shap_explain(self, image):
        try:
            if len(image.shape) == 3:
                # attention inuput de la fonction doit être de la dimension d'un batch (shape: [1, 3, 224, 224])
                shap_values = self.explainer.shap_values(image.unsqueeze(0))
                output = self.model(image.unsqueeze(0))
                predicted_class = output.argmax(dim=1).item()

            elif len(image.shape) == 4:
                shap_values = self.explainer.shap_values(image)
                output = self.model(image)
                predicted_class = output.argmax(dim=1).item()

        except:
            print('Image dimension failed')

        reshaped_shape_values = shap_values[..., predicted_class].squeeze(0)
        reshaped_shape_values_hwc = reshaped_shape_values.transpose(1, 2, 0)
        image_test_hwc = image.numpy().transpose(1, 2, 0)

        return reshaped_shape_values_hwc, image_test_hwc


# class ConsistencyMetrics:
#     def __init__(self, dico_phi, dico_pred, img, label):
#         self.dico_phi = dico_phi
#         self.dico_pred = dico_pred
#         self.img = img
#         self.label = label

#     def compute_metrics(self):
#         c = 0
#         S = {}
#         S_d = {}
#         distances = {}
#         n = len(self.dico_phi.keys())
#         for i in self.dico_phi.keys():
#             c += 1
#             for j in range(c, n+1):
#                 if i == j:
#                     continue
#                 else:
#                     if self.dico_pred[i] == int(self.label) and self.dico_pred[j] == int(self.label):
#                         distances[(i, j)] = euclidean_distances(
#                             self.dico_phi[i], self.dico_phi[j])
#                         S[(i, j)] = float(distances[(i, j)])
#                     else:
#                         distances[(i, j)] = euclidean_distances(
#                             self.dico_phi[i], self.dico_phi[j])
#                         S_d[(i, j)] = float(distances[(i, j)])

#         MeGe_x = 1 / (1 + (1 / len(S.keys()))*np.sum(list(S.values())))

#         return S, S_d, MeGe_x

class ConsistencyMetrics:
    def __init__(self, dico_phi, dico_pred, img, label, dist='euclidean'):
        self.dico_phi = dico_phi
        self.dico_pred = dico_pred
        self.img = img
        self.label = int(label)
        self.dist = dist.lower()  # pour éviter la casse

    def compute_metrics(self):
        S = {}      # Paires avec même prédiction correcte
        S_d = {}    # Paires avec au moins une prédiction incorrecte
        distances = {}

        keys = list(self.dico_phi.keys())
        n = len(keys)

        for i_idx in range(n):
            i = keys[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = keys[j_idx]

                # Reshape SHAP maps en vecteurs ligne
                phi_i = self.dico_phi[i].reshape(-1).reshape(1, -1)
                phi_j = self.dico_phi[j].reshape(-1).reshape(1, -1)

                # Choix de la distance
                if self.dist == 'euclidean':
                    dist = euclidean_distances(phi_i, phi_j)[0][0]
                elif self.dist == 'cosine':
                    dist = cosine_distances(phi_i, phi_j)[0][0]
                else:
                    raise ValueError("dist must be 'euclidean' or 'cosine'")

                distances[(i, j)] = dist

                # Catégorisation
                if self.dico_pred[i] == self.label and self.dico_pred[j] == self.label:
                    S[(i, j)] = dist
                else:
                    S_d[(i, j)] = dist

        # Calcul de MeGe
        if len(S) > 0:
            MeGe_x = 1 / (1 + (1 / len(S)) * np.sum(list(S.values())))
        else:
            MeGe_x = 0.0  # ou np.nan selon ce que tu préfères

        return S, S_d, MeGe_x


def train_single_model(prep, epochs=1):
    train_data = prep.load_dataset(train=True)
    train_loader = prep.create_dataloader(train_data)
    train_data_split1 = prep.split_data(train_data, K=4)[0][0]

    trainer = Trainer(num_classes=10)
    trainer.train(train_data_split1, epochs=epochs, max_batches=3)

    return trainer, train_loader, train_data


def explain_and_plot(trainer, train_loader):
    background = next(iter(train_loader))[0][:10]
    expl = Explainer(trainer.model, background_data=background)

    img = next(iter(train_loader))[0][0]
    shap_map, img_np = expl.shap_explain(img)
    output = trainer.model(img.unsqueeze(0))
    pred = trainer.predict(img.unsqueeze(0))

    print('class = ', pred)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    shap.image_plot(shap_map, img_np)


def run_kfold_training(prep, background, img):
    train_data = prep.load_dataset(train=True)
    dataset_splited_4fold = prep.split_data(train_data, K=4)
    dico_phi = {}
    dico_pred = {}
    c = 1
    for fold in dataset_splited_4fold[0]:
        print(f'\n split {c} ')
        model = Trainer(num_classes=10)
        model.train(fold, epochs=1, max_batches=3)
        expl = Explainer(model.model, background_data=background)
        shap_map, img_np = expl.shap_explain(img)
        dico_phi[c] = shap_map
        predicted_class = model.predict(img.unsqueeze(0))
        dico_pred[c] = predicted_class
        c += 1
    return dico_phi, dico_pred, img, train_data


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)


# if __name__ == "__main__":

#     # Initialisation
#     prep = Preprocessor()
#     train_data = prep.load_dataset(train=True)
#     train_loader = prep.create_dataloader(train_data)

#     # Split data
#     train_data_split1 = prep.split_data(train_data, K=4)[
#         0][0]  # test sur les 3 premiers batch

#     # Modèle
#     trainer = Trainer(num_classes=10)
#     trainer.train(train_data_split1, epochs=1, max_batches=3)

#     # Variables
#     background = next(iter(train_loader))[0][:10]

#     # Préparation SHAP
#     expl = Explainer(trainer.model, background_data=background)

#     # Visualisation SHAP
#     img = next(iter(train_loader))[0][0]
#     shap_map, img_np = expl.shap_explain(img)

#     # Predict class
#     output = trainer.model(img.unsqueeze(0))
#     pred = trainer.predict(img.unsqueeze(0))
#     # pred = trainer.predict(output)
#     print('class = ', pred)

#     # Affichage
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 5))
#     shap.image_plot(shap_map, img_np)

#     # Boucle d'entrainement des K fonctions
#     dataset_splited_4fold = prep.split_data(train_data, K=4)
#     dico_phi = {}
#     dico_pred = {}
#     # Compteur
#     c = 1
#     for fold in dataset_splited_4fold[0]:
#         print(f'\n split {c} ')
#         model = Trainer(num_classes=10)
#         model.train(fold, epochs=1, max_batches=3)
#         expl = Explainer(model.model, background_data=background)
#         shap_map, img_np = expl.shap_explain(img)
#         # Sauvegarde des valeurs shaps pour img pour chaque fonction
#         dico_phi[c] = shap_map
#         predicted_class = model.predict(img.unsqueeze(0))
#         # Sauvgarde des prédictions de chaque fonction
#         dico_pred[c] = predicted_class
#         c += 1

if __name__ == "__main__":
    prep = Preprocessor()
    trainer, train_loader, train_data = train_single_model(prep)

    # Facultatif : afficher l’explication SHAP
    # explain_and_plot(trainer, train_loader)

    background = next(iter(train_loader))[0][:10]
    # img = next(iter(train_loader))[0][0]
    images, labels = next(iter(train_loader))
    img = images[0]
    label = labels[0].item()
    print('\n label img :', label)

    # Juste exécuter la boucle KFold
    dico_phi, dico_pred, img, train_data = run_kfold_training(
        prep, background, img)

    # Calcul le score MeGe et les matrices S
    metrics = ConsistencyMetrics(
        dico_phi, dico_pred, img, label, dist='cosine')
    S, S_d, MeGe_x = metrics.compute_metrics()

    # Afficher l’explication SHAP
    explain_and_plot(trainer, train_loader)
    print('/n MeGe = ', MeGe_x)

    # Supprime le dossier ./data
    delete_folder('./data')
