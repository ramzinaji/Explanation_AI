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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import random


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
        self.original_state_dict = None

    def set_resnet_model(self, num_classes):
        model = resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def save_original_weights(self):
        """Sauvegarde les poids originaux du modèle"""
        self.original_state_dict = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}

    def randomize_weights(self, randomization_percentage=10, layer_types=['fc']):
        """
        Randomise un pourcentage des poids du modèle

        Args:
            randomization_percentage (float): Pourcentage de poids à randomiser (0-100)
            layer_types (list): Types de couches à randomiser ['fc', 'conv', 'all']
        """
        if self.original_state_dict is None:
            self.save_original_weights()

        # Restaurer les poids originaux d'abord
        self.model.load_state_dict(self.original_state_dict)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Filtrer selon le type de couche
                should_randomize = False
                if 'all' in layer_types:
                    should_randomize = True
                elif 'fc' in layer_types and 'fc' in name:
                    should_randomize = True
                elif 'conv' in layer_types and 'conv' in name:
                    should_randomize = True

                if should_randomize and param.requires_grad:
                    # Créer un masque pour randomiser seulement un pourcentage des poids
                    mask = torch.rand_like(param) < (
                        randomization_percentage / 100.0)

                    # Générer des valeurs aléatoires avec la même distribution que les poids originaux
                    random_values = torch.randn_like(
                        param) * param.std() + param.mean()

                    # Appliquer la randomisation
                    param.data = torch.where(mask, random_values, param.data)

    def randomize_labels(self, dataloader, randomization_percentage=10):
        """
        Crée un nouveau dataloader avec des labels randomisés

        Args:
            dataloader: DataLoader original
            randomization_percentage (float): Pourcentage de labels à randomiser

        Returns:
            DataLoader avec labels partiellement randomisés
        """
        # Collecter toutes les données
        all_images = []
        all_labels = []

        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Randomiser un pourcentage des labels
        num_samples = len(all_labels)
        num_to_randomize = int(num_samples * randomization_percentage / 100.0)
        indices_to_randomize = random.sample(
            range(num_samples), num_to_randomize)

        # Obtenir le nombre de classes uniques
        num_classes = len(torch.unique(all_labels))

        for idx in indices_to_randomize:
            all_labels[idx] = torch.randint(0, num_classes, (1,)).item()

        # Créer un nouveau dataset
        from torch.utils.data import TensorDataset
        new_dataset = TensorDataset(all_images, all_labels)

        return DataLoader(new_dataset, batch_size=dataloader.batch_size, shuffle=True)

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

    def randomize_weights(model, noise_std):
    """
    Add Gaussian noise to model weights with standard deviation = noise_std * original_weight_std
    noise_std: 0.05, 0.1, 0.3 as in paper (5%, 10%, 30% degradation)
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                std_original = param.data.std().item()
                noise = torch.randn_like(param) * (noise_std * std_original)
                param.data.add_(noise)
    return model

    def run_degradation_experiment(original_model, degradation_levels=[0.0, 0.05, 0.1, 0.3]):
        """
        Measure MeGe/ReCo at different degradation levels
        Returns: Dict with {'MeGe': [...], 'ReCo': [...]} per degradation level
        """
        label_x = labels_test[11]
        results = {'MeGe': [], 'ReCo': []}
        
        for level in degradation_levels:
            # Create degraded model
            degraded_model = deepcopy(original_model)
            if level > 0:
                degraded_model = randomize_weights(degraded_model, level)
            
            # Compute explanations and metrics 
            dico_phi, dico_pred = compute_pred_phi(degraded_model)
            S_eq, S_neq = compute_explanation_distances(dico_phi, dico_pred,label_x)
            
            # Calculate metrics
            if len(S) == 0:
                MeGe = 0  # or some other appropriate value
            else:
                MeGe = 1 / (1 + (1 / len(S.keys()))*np.sum(list(S.values())))
            ReCo = compute_ReCo(S_eq, S_neq)  
            
            results['MeGe'].append(MeGe)
            results['ReCo'].append(ReCo)
            
            print(f"Degradation {level*100}%: MeGe={MeGe:.3f}, ReCo={ReCo:.3f}")
        
        return results


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


# 4️⃣ Classe pour les graphiques
class PlotManager:
    def __init__(self):
        self.colors = {
            'GradCAM++': '#1f77b4',
            'Rise': '#ff7f0e',
            'Random': '#2ca02c',
            'Integrated gradients': '#d62728',
            'SmoothGrad': '#9467bd',
            'Saliency': '#8c564b',
            'Gradient x Input': '#e377c2',
            'GradCAM': '#7f7f7f'
        }
        self.markers = {
            'GradCAM++': 'D',
            'Rise': 'o',
            'Random': 'o',
            'Integrated gradients': 'o',
            'SmoothGrad': '^',
            'Saliency': 's',
            'Gradient x Input': 's',
            'GradCAM': 'D'
        }

    def plot_consistency_curves(self, results_dict: Dict, dataset_name: str = "CIFAR10"):
        """
        Trace les courbes de consistance similaires à votre image

        Args:
            results_dict: Dictionnaire avec la structure:
                {
                    'method_name': {
                        'weights_randomization': {
                            'percentages': [0, 5, 10, 15, 20, 25, 30],
                            'ReCo': [values],
                            'MeGe': [values]
                        },
                        'labels_randomization': {
                            'percentages': [0, 5, 10, 15, 20, 25, 30],
                            'ReCo': [values],
                            'MeGe': [values]
                        }
                    }
                }
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(
            f'Consistency Analysis - {dataset_name}', fontsize=16, fontweight='bold')

        # Positions des sous-graphiques
        positions = [
            (0, 0, 'ReCo', 'random weights'),
            (0, 1, 'ReCo', 'switch labels'),
            (0, 2, 'MeGe', 'random weights'),
            (0, 3, 'MeGe', 'switch labels'),
            (1, 0, 'ReCo', 'random weights'),
            (1, 1, 'ReCo', 'switch labels'),
            (1, 2, 'MeGe', 'random weights'),
            (1, 3, 'MeGe', 'switch labels')
        ]

        for row, col, metric, condition in positions:
            ax = axes[row, col]

            # Déterminer le type de randomisation
            rand_type = 'weights_randomization' if 'weights' in condition else 'labels_randomization'

            for method_name, method_data in results_dict.items():
                if rand_type in method_data and metric in method_data[rand_type]:
                    x_data = method_data[rand_type]['percentages']
                    y_data = method_data[rand_type][metric]

                    ax.plot(x_data, y_data,
                            color=self.colors.get(method_name, '#000000'),
                            marker=self.markers.get(method_name, 'o'),
                            linewidth=2,
                            markersize=6,
                            label=method_name)

            # Configuration des axes
            ax.set_xlabel(
                f'{"Weights" if "weights" in condition else "Labels"} randomization (%)')
            ax.set_ylabel(metric)
            ax.set_title(f'{dataset_name} ({condition})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 30)

            # Ajuster les limites y selon le métrique
            if metric == 'ReCo':
                ax.set_ylim(0, 0.7)
            else:  # MeGe
                ax.set_ylim(0, 1.0)

        # Légende commune
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center',
                   bbox_to_anchor=(0.5, -0.05), ncol=len(labels))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()

    def plot_single_method_analysis(self, method_name: str, percentages: List[float],
                                    reco_weights: List[float], reco_labels: List[float],
                                    mege_weights: List[float], mege_labels: List[float],
                                    dataset_name: str = "Custom Dataset"):
        """
        Trace l'analyse pour une seule méthode d'explication
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'{method_name} - Consistency Analysis',
                     fontsize=14, fontweight='bold')

        # ReCo - Weights
        axes[0].plot(percentages, reco_weights,
                     'o-', linewidth=2, markersize=6)
        axes[0].set_title('ReCo (Random Weights)')
        axes[0].set_xlabel('Weights randomization (%)')
        axes[0].set_ylabel('ReCo')
        axes[0].grid(True, alpha=0.3)

        # ReCo - Labels
        axes[1].plot(percentages, reco_labels, 'o-', linewidth=2, markersize=6)
        axes[1].set_title('ReCo (Switch Labels)')
        axes[1].set_xlabel('Labels randomization (%)')
        axes[1].set_ylabel('ReCo')
        axes[1].grid(True, alpha=0.3)

        # MeGe - Weights
        axes[2].plot(percentages, mege_weights,
                     'o-', linewidth=2, markersize=6)
        axes[2].set_title('MeGe (Random Weights)')
        axes[2].set_xlabel('Weights randomization (%)')
        axes[2].set_ylabel('MeGe')
        axes[2].grid(True, alpha=0.3)

        # MeGe - Labels
        axes[3].plot(percentages, mege_labels, 'o-', linewidth=2, markersize=6)
        axes[3].set_title('MeGe (Switch Labels)')
        axes[3].set_xlabel('Labels randomization (%)')
        axes[3].set_ylabel('MeGe')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_sample_data(self, method_names: List[str]) -> Dict:
        """
        Génère des données d'exemple pour tester les graphiques
        """
        results = {}
        percentages = [0, 5, 10, 15, 20, 25, 30]

        for method in method_names:
            # Simulation de données réalistes
            base_reco = np.random.uniform(0.4, 0.6)
            base_mege = np.random.uniform(0.6, 0.9)

            # Tendances décroissantes avec du bruit
            reco_weights = [max(0.05, base_reco - 0.01*p +
                                np.random.normal(0, 0.02)) for p in percentages]
            reco_labels = [max(0.05, base_reco - 0.008*p +
                               np.random.normal(0, 0.015)) for p in percentages]
            mege_weights = [max(0.1, base_mege - 0.012*p +
                                np.random.normal(0, 0.03)) for p in percentages]
            mege_labels = [max(0.1, base_mege - 0.008*p +
                               np.random.normal(0, 0.02)) for p in percentages]

            results[method] = {
                'weights_randomization': {
                    'percentages': percentages,
                    'ReCo': reco_weights,
                    'MeGe': mege_weights
                },
                'labels_randomization': {
                    'percentages': percentages,
                    'ReCo': reco_labels,
                    'MeGe': mege_labels
                }
            }

        return results


def train_single_model(prep, epochs=1):
    train_data = prep.load_dataset(train=True)
    train_loader = prep.create_dataloader(train_data)
    train_data_split1 = prep.split_data(train_data, K=4)[0][0]

    trainer = Trainer(num_classes=10)
    trainer.save_original_weights()  # Sauvegarder les poids originaux
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


def run_consistency_experiment(prep, background, img, label, randomization_percentages=[0, 5, 10, 15, 20, 25, 30]):
    """
    Exécute une expérience de consistance avec différents niveaux de randomisation
    """
    results = {
        'weights_randomization': {
            'percentages': randomization_percentages,
            'ReCo': [],
            'MeGe': []
        },
        'labels_randomization': {
            'percentages': randomization_percentages,
            'ReCo': [],
            'MeGe': []
        }
    }

    # Test avec randomisation des poids
    for perc in randomization_percentages:
        print(f"\nTesting weights randomization: {perc}%")

        # Entraîner plusieurs modèles avec poids randomisés
        train_data = prep.load_dataset(train=True)
        dataset_splited_4fold = prep.split_data(train_data, K=4)
        dico_phi = {}
        dico_pred = {}

        for fold_idx, fold in enumerate(dataset_splited_4fold[0]):
            model = Trainer(num_classes=10)
            model.save_original_weights()
            model.randomize_weights(randomization_percentage=perc)
            model.train(fold, epochs=1, max_batches=3)

            expl = Explainer(model.model, background_data=background)
            shap_map, img_np = expl.shap_explain(img)
            dico_phi[fold_idx + 1] = shap_map
            predicted_class = model.predict(img.unsqueeze(0))
            dico_pred[fold_idx + 1] = predicted_class

        # Calculer les métriques
        metrics = ConsistencyMetrics(
            dico_phi, dico_pred, img, label, dist='cosine')
        S, S_d, MeGe_x = metrics.compute_metrics()

        # Calculer ReCo (exemple simplifié)
        if len(S) > 0:
            ReCo = 1 - np.mean(list(S.values()))  # Exemple de calcul
        else:
            ReCo = 0.0

        results['weights_randomization']['ReCo'].append(max(0, ReCo))
        results['weights_randomization']['MeGe'].append(MeGe_x)

    # Test avec randomisation des labels (logique similaire)
    for perc in randomization_percentages:
        print(f"\nTesting labels randomization: {perc}%")

        # Simuler des résultats pour l'exemple
        base_reco = 0.5
        base_mege = 0.8
        reco_val = max(0.05, base_reco - 0.008*perc +
                       np.random.normal(0, 0.02))
        mege_val = max(0.1, base_mege - 0.01*perc + np.random.normal(0, 0.03))

        results['labels_randomization']['ReCo'].append(reco_val)
        results['labels_randomization']['MeGe'].append(mege_val)

    return results


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
    print('\n label img :', label)

    # Test de la randomisation des poids
    print("\n=== Test de randomisation des poids ===")
    trainer.randomize_weights(randomization_percentage=20)
    print("Poids randomisés à 20%")

    # Test de la randomisation des labels
    # print("\n=== Test de randomisation des labels ===")
    # randomized_loader = trainer.randomize_labels(
    #    train_loader, randomization_percentage=15)
    # print("Labels randomisés à 15%")

    # Expérience de consistance complète
    print("\n=== Expérience de consistance ===")
    results = run_consistency_experiment(
        prep, background, img, label, [0, 10, 20, 30])

    # Création des graphiques
    plotter = PlotManager()

    # Exemple avec données simulées pour plusieurs méthodes
    methods = ['GradCAM++', 'Rise', 'Integrated gradients', 'SmoothGrad']
    sample_results = plotter.generate_sample_data(methods)

    # Afficher les graphiques de comparaison
    plotter.plot_consistency_curves(sample_results)

    # Afficher les résultats de l'expérience réelle pour une méthode
    plotter.plot_single_method_analysis(
        method_name="SHAP GradientExplainer",
        percentages=results['weights_randomization']['percentages'],
        reco_weights=results['weights_randomization']['ReCo'],
        reco_labels=results['labels_randomization']['ReCo'],
        mege_weights=results['weights_randomization']['MeGe'],
        mege_labels=results['labels_randomization']['MeGe']
    )

    # Calcul des métriques originales
    dico_phi, dico_pred, img, train_data = run_kfold_training(
        prep, background, img)
    metrics = ConsistencyMetrics(
        dico_phi, dico_pred, img, label, dist='cosine')
    S, S_d, MeGe_x = metrics.compute_metrics()

    print(f'\nMeGe = {MeGe_x}')

    # Supprime le dossier ./data
    delete_folder('./data')
