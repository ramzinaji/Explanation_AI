import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.model_selection import KFold
import numpy as np
import shap
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
import os
import matplotlib.pyplot as plt


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

    def randomize_weights(self, noise_std):
        """
        Add Gaussian noise to model weights with standard deviation = noise_std * original_weight_std
        noise_std: 0.05, 0.1, 0.3 as in paper (5%, 10%, 30% degradation)
        """
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    std_original = param.data.std().item()
                    noise = torch.randn_like(
                        param) * (noise_std * std_original)
                    param.data.add_(noise)
        return self.model


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
                # attention input de la fonction doit être de la dimension d'un batch (shape: [1, 3, 224, 224])
                shap_values = self.explainer.shap_values(image.unsqueeze(0))
                output = self.model(image.unsqueeze(0))
                predicted_class = output.argmax(dim=1).item()

            elif len(image.shape) == 4:
                shap_values = self.explainer.shap_values(image)
                output = self.model(image)
                predicted_class = output.argmax(dim=1).item()

        except Exception as e:
            print(f'Image dimension failed: {e}')
            return None, None

        reshaped_shape_values = shap_values[..., predicted_class].squeeze(0)
        reshaped_shape_values_hwc = reshaped_shape_values.transpose(1, 2, 0)
        image_test_hwc = image.numpy().transpose(1, 2, 0)

        return reshaped_shape_values_hwc, image_test_hwc


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

    def compute_ReCo(self, S, S_d):
        """
        Compute Relative Consistency (ReCo) given:
        - S:   Distances when both models agree (correct predictions)
        - S_d: Distances when models disagree (one correct, one wrong)

        Returns:
        - ReCo score (higher = more consistent explanations)
        """
        if len(S) == 0 or len(S_d) == 0:
            return 0.0  # Edge case: no comparable samples

        all_dists = np.array(list(S.values()) + list(S_d.values()))
        labels = np.array([0]*len(S) + [1]*len(S_d))

        best_balanced_acc = 0

        for gamma in np.unique(all_dists):
            preds = (all_dists <= gamma).astype(int)

            # TPR: True Positive Rate (sensibilité)
            if np.sum(labels == 0) > 0:
                tpr = np.sum((preds == 0) & (labels == 0)) / \
                    np.sum(labels == 0)
            else:
                tpr = 0

            # TNR: True Negative Rate (spécificité)
            if np.sum(labels == 1) > 0:
                tnr = np.sum((preds == 1) & (labels == 1)) / \
                    np.sum(labels == 1)
            else:
                tnr = 0

            balanced_acc = tpr + tnr - 1

            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc

        return max(best_balanced_acc, 0)

    def run_degradation_experiment(self, prep, background, degradation_levels=[0.0, 0.05, 0.1, 0.3]):
        """
        Measure MeGe/ReCo at different degradation levels
        Returns: Dict with {'MeGe': [...], 'ReCo': [...]} per degradation level
        """
        results = {'MeGe': [], 'ReCo': []}

        for level in degradation_levels:
            print(f"\nTesting degradation level: {level*100}%")

            # Entraîner de nouveaux modèles avec ce niveau de dégradation
            train_data = prep.load_dataset(train=True)
            dataset_splited_4fold = prep.split_data(train_data, K=4)

            dico_phi_deg = {}
            dico_pred_deg = {}

            for c, fold in enumerate(dataset_splited_4fold[0], 1):
                model = Trainer(num_classes=10)
                model.train(fold, epochs=1, max_batches=3)

                # Appliquer la dégradation si nécessaire
                if level > 0:
                    model.randomize_weights(level)

                expl = Explainer(model.model, background_data=background)
                shap_map, _ = expl.shap_explain(self.img)

                if shap_map is not None:
                    dico_phi_deg[c] = shap_map
                    predicted_class = model.predict(self.img.unsqueeze(0))
                    dico_pred_deg[c] = predicted_class

            # Calculer les métriques pour ce niveau de dégradation
            if len(dico_phi_deg) > 1:  # Besoin d'au moins 2 modèles pour calculer les distances
                metrics_deg = ConsistencyMetrics(
                    dico_phi_deg, dico_pred_deg, self.img, self.label, dist=self.dist)

                S_eq, S_neq, MeGe = metrics_deg.compute_metrics()
                ReCo = metrics_deg.compute_ReCo(S_eq, S_neq)

                results['MeGe'].append(MeGe)
                results['ReCo'].append(ReCo)

                print(
                    f"Degradation {level*100}%: MeGe={MeGe:.3f}, ReCo={ReCo:.3f}")
            else:
                results['MeGe'].append(0.0)
                results['ReCo'].append(0.0)
                print(f"Degradation {level*100}%: Insufficient models trained")

        return results


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

    if shap_map is not None and img_np is not None:
        output = trainer.model(img.unsqueeze(0))
        pred = trainer.predict(img.unsqueeze(0))

        print('class = ', pred)

        plt.figure(figsize=(8, 5))
        shap.image_plot(shap_map, img_np)


def run_kfold_training(prep, background, img, K=4, batch=3):
    train_data = prep.load_dataset(train=True)
    dataset_splited_4fold = prep.split_data(train_data, K=K)
    dico_phi = {}
    dico_pred = {}
    c = 1
    for fold in dataset_splited_4fold[0]:
        print(f'\n split {c} ')
        model = Trainer(num_classes=10)
        model.train(fold, epochs=1, max_batches=batch)
        expl = Explainer(model.model, background_data=background)
        shap_map, img_np = expl.shap_explain(img)

        if shap_map is not None:
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


if __name__ == "__main__":
    prep = Preprocessor()
    trainer, train_loader, train_data = train_single_model(prep)

    background = next(iter(train_loader))[0][:10]
    images, labels = next(iter(train_loader))
    img = images[0]
    label = labels[0].item()
    true_class_name = train_data.classes[label]

    print(f'\n label img and name: {label}, {true_class_name}')

    # Exécuter la boucle KFold pour obtenir les métriques de base
    dico_phi, dico_pred, img, train_data = run_kfold_training(
        prep, background, img, batch=50)

    # Calculer le score MeGe et les matrices S
    metrics = ConsistencyMetrics(
        dico_phi, dico_pred, img, label, dist='cosine')
    S, S_d, MeGe_x = metrics.compute_metrics()

    # Afficher l'explication SHAP
    explain_and_plot(trainer, train_loader)
    print('\n dico pred :', dico_pred)
    print('\n MeGe = ', MeGe_x, '\n S = ', S, '\n S_d = ', S_d)

    # Expérience de dégradation
    print("\n" + "="*50)
    print("Starting degradation experiment...")
    print("="*50)

    degradation_levels = [0.0, 0.05, 0.1, 0.3]
    results = metrics.run_degradation_experiment(
        prep, background, degradation_levels)

    print("\nResults at different degradation levels:")
    for i, level in enumerate(degradation_levels):
        print(
            f"Degradation {level*100:4.1f}%: MeGe={results['MeGe'][i]:.3f}, ReCo={results['ReCo'][i]:.3f}")

    # Plot results (as in paper Figure 3)
    plt.figure(figsize=(10, 6))
    plt.plot(degradation_levels,
             results['MeGe'], marker='o', label='MeGe', linewidth=2, markersize=8)
    plt.plot(degradation_levels,
             results['ReCo'], marker='s', label='ReCo', linewidth=2, markersize=8)
    plt.xlabel('Weight degradation level', fontsize=12)
    plt.ylabel('Metric value', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Sanity Check: Metric Sensitivity to Model Degradation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Supprime le dossier ./data
    delete_folder('./data')
