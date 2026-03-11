# 📊 Système de Monitoring ML - Documentation

Ce projet implémente un système complet de monitoring pour l'API de prédiction de risque de crédit.

## 🎯 Objectifs

- ✅ API fonctionnelle avec FastAPI
- ✅ Logging structuré de toutes les requêtes
- ✅ Stockage en base de données SQLite
- ✅ Dashboard de monitoring avec Streamlit
- ✅ Détection automatique de Data Drift avec Evidently

## 📁 Structure du Projet

```
.
├── api/
│   ├── main.py                    # Application FastAPI principale
│   ├── database/
│   │   ├── database.py            # Configuration SQLite
│   │   └── db_models.py           # Modèles SQLAlchemy
│   ├── models/
│   │   ├── schemas.py             # Schémas Pydantic
│   │   └── model_loader.py        # Chargement du modèle ML
│   ├── routers/
│   │   ├── prediction.py          # Endpoints de prédiction
│   │   └── monitoring.py          # Endpoints de monitoring
│   └── monitoring/
│       └── drift_detection.py     # Détection de drift avec Evidently
├── dashboard/
│   └── monitoring_dashboard.py    # Dashboard Streamlit
├── start_api.py                   # Script de démarrage API
├── start_dashboard.py             # Script de démarrage Dashboard
├── test_api.py                    # Script de test
└── logs.db                        # Base de données (créée au démarrage)
```

## 🔧 Installation

1. **Installer les dépendances supplémentaires:**

```bash
pip install streamlit==1.31.0 plotly==5.18.0 evidently==0.4.14
```

Ou installer depuis le fichier:

```bash
pip install -r requirements_new.txt
```

2. **Vérifier que toutes les dépendances existantes sont installées:**

```bash
pip install -r requirements.txt
```

## 🚀 Démarrage

### Option 1: Démarrage Manuel

**Terminal 1 - API:**

```bash
python start_api.py
```

L'API sera accessible sur: http://localhost:8000

**Terminal 2 - Dashboard:**

```bash
python start_dashboard.py
```

Le dashboard sera accessible sur: http://localhost:8501

### Option 2: Démarrage avec uvicorn/streamlit directement

**API:**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Dashboard:**

```bash
streamlit run dashboard/monitoring_dashboard.py --server.port 8501
```

## 📊 Ce qui est Loggé

### A. Traçabilité

- `request_id`: Identifiant unique de la requête
- `timestamp`: Date et heure de la prédiction
- `model_version`: Version du modèle utilisé
- `api_version`: Version de l'API
- `environment`: Environnement (dev/prod)

### B. Inputs (pour Data Drift)

- `input_features`: Toutes les features envoyées (JSON)
- `n_features`: Nombre de features
- `missing_values_count`: Nombre de valeurs manquantes
- `schema_version`: Version du schéma de données

### C. Outputs Modèle

- `prediction_score`: Score de prédiction (probabilité)
- `prediction_class`: Classe prédite (0 ou 1)
- `threshold`: Seuil utilisé pour la classification

### D. Métriques Techniques

- `status_code`: Code HTTP de la réponse
- `latency_ms`: Latence totale de la requête
- `inference_time_ms`: Temps d'inférence du modèle
- `error_message`: Message d'erreur (si applicable)
- `error_type`: Type d'erreur (si applicable)

## 🔌 Endpoints de l'API

### Health Check

```bash
GET /health
```

Retourne le statut de l'API et du modèle.

### Prédiction

```bash
POST /api/v1/predict
```

**Body:**

```json
{
  "features": {
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637
  }
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-03-02T10:30:00Z",
  "prediction_score": 0.23,
  "prediction_class": 0,
  "threshold": 0.5,
  "model_version": "LogisticRegression_v1.0",
  "api_version": "1.0.0",
  "inference_time_ms": 12.5
}
```

### Statistiques

```bash
GET /api/v1/stats
```

Retourne les statistiques de monitoring (latence, taux d'erreur, etc.)

### Drift Summary

```bash
GET /api/v1/monitoring/drift/summary
```

Retourne un résumé du drift détecté.

### Générer Rapport de Drift

```bash
POST /api/v1/monitoring/drift/report?output_path=reports/drift_report.html
```

Génère un rapport HTML complet avec Evidently.

## 📈 Dashboard de Monitoring

Le dashboard affiche:

### Monitoring API

- ✅ Latence moyenne + P95
- ✅ Taux d'erreur
- ✅ Volume de requêtes
- ✅ Distribution des codes HTTP

### Monitoring ML

- ✅ Distribution des scores de prédiction
- ✅ Évolution du score dans le temps
- ✅ Distribution des classes prédites
- ✅ Temps d'inférence

### Détails Techniques

- Version du modèle
- Statistiques de performance (P50, P95, P99)
- Dernières prédictions

## 🧪 Tester l'API

Un script de test est fourni:

```bash
python test_api.py
```

Ce script va:

1. Vérifier le health check
2. Faire une prédiction de test
3. Lancer 5 prédictions successives
4. Afficher les statistiques
5. Vérifier le drift summary

## 🔍 Data Drift avec Evidently

### Configurer le Dataset de Référence

1. **Charger un dataset de référence:**

```bash
POST /api/v1/monitoring/drift/set-reference
{
  "data_path": "data/dataset_final.csv"
}
```

2. **Obtenir un résumé du drift:**

```bash
GET /api/v1/monitoring/drift/summary
```

Retourne:
- Nombre d'échantillons en référence et en production
- Comparaison des distributions par feature
- Pourcentage de drift détecté
- Liste des features qui ont dérivé

3. **Générer un rapport HTML complet:**

```bash
POST /api/v1/monitoring/drift/report
```

Le rapport sera sauvegardé dans `reports/drift_report.html` avec des visualisations interactives.

### Fonctionnalités de Détection de Drift

- ✅ Comparaison automatique avec le dataset de référence
- ✅ Détection du drift par feature (seuil: écart > 10%)
- ✅ Comparaison des moyennes et écarts-types
- ✅ Métriques de qualité des données
- ✅ Rapports HTML interactifs avec Evidently
- ✅ Extraction automatique des données de production depuis les logs

## 📝 Fichiers de Configuration

### .env.example

Copiez ce fichier en `.env` pour configurer l'environnement:

```bash
cp .env.example .env
```

Variables disponibles:

- `API_VERSION`: Version de l'API
- `ENVIRONMENT`: dev ou prod
- `DEFAULT_THRESHOLD`: Seuil de classification (0.5 par défaut)
- `DATABASE_URL`: URL de la base de données
- `REFERENCE_DATA_PATH`: Chemin vers le dataset de référence

## 🐳 Docker et Déploiement

### Construction des images Docker

**API:**
```bash
docker build -t mlops-api -f Dockerfile .
```

**Dashboard:**
```bash
docker build -t mlops-dashboard -f Dockerfile.dashboard .
```

### Démarrage avec Docker Compose

```bash
docker-compose up -d
```

Cela démarre:
- API sur http://localhost:8000
- Dashboard sur http://localhost:8501

### Arrêt des services

```bash
docker-compose down
```

## 📊 Analyse des Performances

Un script d'analyse des performances est disponible:

```bash
python analyse_performances.py
```

Ce script analyse:
- **Latence de l'API** (moyenne, P95, P99, max)
- **Temps d'inférence du modèle**
- **Taux d'erreur**
- **Distribution des scores de prédiction**
- **Volume de requêtes par heure**
- **Identification des goulots d'étranglement**

Le script fournit également des recommandations pour optimiser les performances.

## 🔄 CI/CD avec GitHub Actions

Le pipeline CI/CD est configuré dans [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml).

### Configuration requise

Dans les secrets GitHub, ajouter:
- `DOCKER_USERNAME`: Nom d'utilisateur Docker Hub
- `DOCKER_PASSWORD`: Mot de passe Docker Hub

### Le pipeline effectue:

1. **Tests automatiques** à chaque push/PR
2. **Build des images Docker** sur push vers main
3. **Push des images** vers Docker Hub

## 🎓 Choix Techniques

### Architecture

- **FastAPI**: Choisi pour sa performance et sa documentation automatique (Swagger)
- **SQLite**: Base simple pour le logging, peut être remplacée par PostgreSQL en production
- **Streamlit**: Dashboard interactif simple à déployer
- **Evidently**: Détection de drift robuste et rapports visuels

### Chargement du modèle

Le modèle est chargé **une seule fois au démarrage** de l'API (dans la fonction `lifespan`), ce qui évite de recharger le modèle à chaque requête et améliore considérablement les performances.

### Logging structuré

Tous les logs contiennent:
- Traçabilité complète (request_id, timestamp)
- Inputs et outputs pour analyse du drift
- Métriques techniques (latence, temps d'inférence)

Cela permet une analyse précise des performances et du comportement du modèle en production.

## 🎓 Prochaines Étapes

### Optimisations possibles

- [ ] Ajouter un cache Redis pour les prédictions fréquentes
- [ ] Implémenter le batching pour les requêtes multiples
- [ ] Ajouter des alertes automatiques (drift, erreurs)
- [ ] Améliorer le dashboard (filtres, exports)

### Production

- [ ] Remplacer SQLite par PostgreSQL
- [ ] Ajouter l'authentification (OAuth2/JWT)
- [ ] Monitoring avec Prometheus/Grafana
- [ ] Mise en place de rate limiting
- [ ] A/B testing de modèles

## 📚 Documentation de l'API

Une fois l'API lancée, consultez la documentation interactive:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🆘 Troubleshooting

### L'API ne démarre pas

- Vérifiez que le port 8000 n'est pas déjà utilisé
- Vérifiez que toutes les dépendances sont installées
- Vérifiez les logs d'erreur

### Le modèle n'est pas chargé

- Vérifiez que le dossier `mlruns` existe
- Vérifiez que le fichier `run_metadata.json` est présent
- Vérifiez les logs au démarrage de l'API

### Le dashboard est vide

- Effectuez d'abord des prédictions via l'API
- Vérifiez que le fichier `logs.db` existe
- Rafraîchissez le dashboard

### Erreur de drift

- Assurez-vous d'avoir défini un dataset de référence
- Vérifiez que les colonnes correspondent
- Assurez-vous d'avoir au moins quelques prédictions en production

## 📞 Support

Pour toute question ou problème, consultez:

- La documentation API: http://localhost:8000/docs
- Les logs de l'API dans le terminal
- Le fichier `logs.db` pour les données

---

🎉 **Félicitations! Votre système de monitoring ML est opérationnel!** 🎉
