# Explanation_AI

Version de python 3.10.16

Le fichier util.py contient toutes les méthodes.
Il est composé de 4 classes :

Preprocessor : importe les données et initialise les dataloaders

Trainer : prend un modèle ResNet et réentraîne une nouvelle dernière couche pour s’adapter aux données d'entraînement

Explainer : importe la fonction d'explicabilité SHAP (notée φ dans le papier de recherche)

ConsistencyMetrics : ensemble des métriques du papier de recherche (MeGe)

Le fichier script.py est utilisé pour le developpement des nouvelles features avant de les intégrer dans le util.py

Les fonction run_kfold_training permet d'entrainer K fonctions, elle prend en paramètre le nombre folds et le nombre de bacth d'entrainement. Le nombre d'époque
fixé a 1 dans cette fonction pour limiter le temps de calcul. 

Pour l'entrainement du modèle , le nombre de batch a été fixé à 50 ce qui peut prendre du temps, pour vérifier que le code fonctionne, on peut fixer le nombre de à 3 dans la fonction run_kfold_training()

