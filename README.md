# Projet Machine Learning avec Streamlit

Ce projet est une application interactive développée avec **Streamlit**, qui illustre un pipeline complet de Machine Learning (ou Deep Learning). L'application se compose de plusieurs étapes : traitement des données, visualisation, modélisation, et évaluation.

#### Documentation utilisateur

Vous retrouverez la documentation utilisateur juste ➡️ [ici](https://github.com/darssn/projet_ML/blob/main/Doc%20User%20-%20Projet%20ML.pdf)
---

## 📋 Fonctionnalités

- **Traitement des données** :
  - Chargement de fichiers CSV.
  - Analyse descriptive (statistiques, données manquantes, standardisation, etc.).
  - Interactions utilisateur pour sélectionner et transformer les données.
  
- **Visualisation** :
  - Graphiques interactifs avec **Altair** (distribution, corrélation, etc.).

- **Modélisation** :
  - Choix d’algorithmes comme Random Forest ou Régression Logistique.
  - Séparation des ensembles d'entraînement et de test.

- **Évaluation** :
  - Affichage de métriques comme la précision et le rapport de classification.

---

## 🛠️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/darssn/projet_ML.git
cd projet_ML
```

### 2. Créer un environnement virtuel

Si vous n'avez pas Virtualenv sur votre machine le télécharger avec la commande suivante : 
```
- pip install virtualenv
```
Si vous l'avez déjà installé : 

```
- python -m venv env
- source env/bin/activate  # Sur Linux/Mac
- env\Scripts\activate     # Sur Windows
```


### 3. Installer les dépendances 

Assurez vous que `pip` est à jour : 

```
pip install --upgrade pip
```

Ensuite, installez les dépendances à partir du fichier `requirements.txt` :
```
pip install -r requirements.txt
```


## 🚀 Utilisation

#### 1. Lancer l'application streamlit : 
```
streamlit run app.py
```

#### 2. Ouvrir l'URL fournie dans votre navigateur (par défaut : http://localhost:8501)
#### 3. Naviguer à travers les onglets pour traiter les données, visualiser les résultats, et entraîner des modèles.




## 📂 Organisation des fichiers

- app.py : Fichier principal de l'application Streamlit.
- requirements.txt : Liste des bibliothèques nécessaires.
- vin.csv : Jeu de données utilisé pour le projet.


## ✍️ Auteurs
- Fatima-Zahra ZABAKA
- Darcy NGUYEN
- Victor LEBRETON 
