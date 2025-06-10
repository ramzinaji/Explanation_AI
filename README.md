# Explanation_AI

Version de python 3.10.16

Le fichier util.py contient toutes les méthodes.
Il est composé de 4 classes :

Preprocessor : importe les données et initialise les dataloaders

Trainer : prend un modèle ResNet et réentraîne une nouvelle dernière couche pour s’adapter aux données d'entraînement

Explainer : importe la fonction d'explicabilité SHAP (notée φ dans le papier de recherche)

ConsistencyMetrics : ensemble des métriques du papier de recherche (MeGe)

Le fichier script.py est utilisé pour les tests.

À faire :
Coder le score ReCo du papier et l’ajouter dans la classe ConsistencyMetrics

Implémenter l’aléa (weight randomization) dans le modèle de la classe Trainer pour pouvoir tracer l’évolution des scores MeGe et ReCo en fonction du taux d’aléa dans les poids

