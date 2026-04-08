# Projet MLOps - Prédiction du risque de crédit

Ce projet implémente une solution complète de MLOps pour la prédiction du risque de crédit. Il inclut un pipeline de traitement des données, des modèles de machine learning, une API de prédiction en temps réel, un système de monitoring et un dashboard de visualisation.

## Vue d'ensemble

Le projet utilise les données du concours Kaggle "Home Credit Default Risk" pour prédire la probabilité qu'un client fasse défaut sur son crédit. L'architecture suit les bonnes pratiques MLOps avec un système de versioning des modèles, un monitoring de la dérive des données et un déploiement automatisé.

## Structure du projet

```
.
├── data/                           # Données brutes et traitées
│   ├── raw/                       # Données sources
│   ├── processed/                 # Données après preprocessing
│   └── reference/                 # Données de référence pour la détection de drift
├── notebooks/                      # Notebooks d'exploration et d'entraînement
│   └── main.ipynb                 # Notebook principal d'entraînement
├── src/                           # Code source de l'application
│   ├── api/                       # API FastAPI
│   │   ├── main.py               # Point d'entrée de l'API
│   │   ├── routers/              # Routes de l'API (prédiction, monitoring)
│   │   ├── models/               # Chargement des modèles et schémas
│   │   ├── database/             # Gestion de la base de données SQLite
│   │   └── monitoring/           # Détection de drift et monitoring
│   ├── dashboard/                 # Dashboard Streamlit
│   │   └── monitoring_dashboard.py
│   └── ml/                        # Modules ML (si nécessaire)
├── scripts/                       # Scripts utilitaires
│   ├── deployment/               # Scripts de déploiement
│   ├── analysis/                 # Scripts d'analyse de performances
│   ├── benchmarks/               # Scripts de benchmarking
│   └── profiling/                # Scripts de profiling
├── tests/                         # Tests unitaires et d'intégration
│   ├── unit/                     # Tests unitaires
│   └── integration/              # Tests d'intégration
├── docker/                        # Configuration Docker
│   └── docker-compose.yml        # Orchestration des services
├── .github/workflows/             # CI/CD GitHub Actions
│   └── ci-cd.yml                 # Pipeline de tests et déploiement
├── mlruns/                        # Artefacts MLflow (modèles, métriques)
├── models/                        # Modèles sauvegardés
├── reports/                       # Rapports de monitoring et de performances
└── requirements.txt               # Dépendances Python
```

## Fonctionnalités principales

### 1. Entraînement des modèles

Le projet utilise plusieurs algorithmes de machine learning pour optimiser la prédiction :

- LightGBM
- XGBoost
- Régression Logistique

L'entraînement est réalisé via le notebook principal avec tracking MLflow pour versionner les modèles et conserver les métriques de performance.

### 2. API de prédiction

API REST développée avec FastAPI qui expose deux endpoints principaux :

**Prédiction de risque**

- Endpoint : `/api/v1/predict`
- Accepte les données client en entrée
- Retourne la probabilité de défaut de paiement et une décision d'acceptation/rejet

**Monitoring**

- Endpoint : `/api/v1/monitoring/stats`
- Fournit des statistiques sur les prédictions
- Détecte la dérive des données (data drift)

### 3. Système de monitoring

Le système de monitoring suit en temps réel :

- Les prédictions effectuées et leur distribution
- La dérive statistique des features (détection de drift)
- Les performances de l'API (temps de réponse, nombre de requêtes)
- Les logs d'erreurs et d'événements

La détection de drift utilise la bibliothèque Evidently pour comparer les données de production avec les données de référence.

### 4. Dashboard de visualisation

Dashboard développé avec Streamlit qui affiche :

- Les métriques clés du modèle en production
- L'évolution des prédictions dans le temps
- Les alertes de drift détectées
- Les statistiques de performances de l'API

### 5. Tests automatisés

Suite de tests complète incluant :

- Tests unitaires des composants individuels (modèle, API, détection de drift)
- Tests d'intégration de bout en bout
- Couverture de code minimale de 60%

Les tests sont exécutés automatiquement via GitHub Actions à chaque push.

### 6. Déploiement containerisé

Le projet utilise Docker pour garantir la reproductibilité :

- Image pour l'API de prédiction
- Image pour le dashboard de monitoring
- Orchestration avec docker-compose pour déployer l'ensemble des services

## Installation et configuration

### Prérequis

- Python 3.11
- Docker et Docker Compose (pour le déploiement)
- Git

### Installation locale

1. Cloner le repository

```bash
git clone <repository-url>
cd OC_P8_MLOps
```

2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine avec les configurations nécessaires.

## Utilisation

### Entraînement du modèle

Exécuter le notebook principal pour entraîner un nouveau modèle :

```bash
jupyter notebook notebooks/main.ipynb
```

Le modèle sera automatiquement enregistré dans MLflow et disponible pour l'API.

### Lancer l'API localement

```bash
python scripts/deployment/start_api.py
```

L'API sera accessible sur `http://localhost:8000`. La documentation interactive est disponible sur `http://localhost:8000/docs`.

### Lancer le dashboard

```bash
python scripts/deployment/start_dashboard.py
```

Le dashboard sera accessible sur `http://localhost:8501`.

### Déploiement avec Docker

Pour déployer l'ensemble de l'application avec Docker Compose :

```bash
cd docker
docker-compose up -d
```

Cette commande démarre les services API et Dashboard en arrière-plan.

## Tests

Exécuter tous les tests :

```bash
pytest
```

Exécuter les tests avec rapport de couverture :

```bash
pytest --cov=src --cov-report=html
```

Les tests d'intégration peuvent être exclus avec :

```bash
pytest -m "not integration"
```

## CI/CD

Le projet utilise GitHub Actions pour l'intégration continue. Le pipeline exécute automatiquement :

1. Installation des dépendances
2. Exécution des tests avec couverture minimale de 60%
3. Génération des rapports de couverture
4. Build des images Docker (sur la branche main)

La configuration du pipeline se trouve dans `.github/workflows/ci-cd.yml`.

## Monitoring et maintenance

### Logs

Les logs de l'API sont stockés dans une base de données SQLite (`logs.db`) et incluent :

- Horodatage de chaque requête
- Données d'entrée et résultats de prédiction
- Métriques de performance

### Détection de drift

Le système vérifie automatiquement la dérive des données à chaque prédiction. En cas de drift détecté, des alertes sont générées et visibles dans le dashboard.

### Analyse des performances

Des scripts d'analyse sont disponibles dans le dossier `scripts/analysis/` pour évaluer :

- Les temps de réponse de l'API
- La distribution des prédictions
- L'évolution des performances du modèle

## Technologies utilisées

**Machine Learning**

- scikit-learn : preprocessing et modèles de base
- LightGBM et XGBoost : modèles de gradient boosting
- MLflow : tracking et versioning des modèles

**Backend**

- FastAPI : framework pour l'API REST
- SQLAlchemy : ORM pour la base de données
- Uvicorn : serveur ASGI

**Frontend**

- Streamlit : dashboard de monitoring
- Plotly : visualisations interactives

**Monitoring**

- Evidently : détection de drift

**Infrastructure**

- Docker : containerisation
- GitHub Actions : CI/CD
- pytest : framework de tests

## Performances

Le modèle en production affiche les performances suivantes sur le jeu de test :

- AUC-ROC : à vérifier dans MLflow
- Temps de réponse moyen de l'API : < 100ms

## Améliorations futures

- Ajouter des tests de charge pour valider la scalabilité
- Implémenter le réentraînement automatique du modèle
- Intégrer des alertes par email en cas de drift critique
- Déployer sur un cloud provider (AWS, GCP, Azure)
- Ajouter l'authentification et l'autorisation à l'API
