import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import json
import os


class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_name = None
        self.feature_names = None
        self.n_features = None
        self.model_path = None  # Pour retrouver les features

    def load_model_from_mlflow(self, run_id: Optional[str] = None) -> bool:
        try:
            if run_id:
                # Charger depuis un run spécifique
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = run_id
            else:
                # Charger le dernier modèle
                # On va chercher dans le dossier mlruns
                model_path = self._find_latest_model()

                if model_path:
                    self.model_path = model_path  # Sauvegarder le chemin
                    # Si c'est un fichier .pkl, charger directement
                    if model_path.endswith('.pkl'):
                        import joblib
                        self.model = joblib.load(model_path)
                        self.model_version = "latest_from_mlruns"
                    else:
                        # Sinon essayer avec MLflow
                        self.model = mlflow.sklearn.load_model(model_path)
                        self.model_version = "latest"
                else:
                    print("No model found in mlruns")
                    return False

            self.model_name = type(self.model).__name__

            # Fixer les attributs manquants pour sklearn ancien
            self._fix_sklearn_compatibility()

            # Charger les métadonnées si disponibles
            self._load_metadata()

            # Extraire les noms de features du modèle
            self._extract_feature_names()

            print(f"Model loaded: {self.model_name} (version: {self.model_version})")
            print(f"Expected features: {len(self.feature_names) if self.feature_names else 'unknown'}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def load_model_from_file(self, model_path: str) -> bool:
        try:
            import joblib
            self.model = joblib.load(model_path)
            self.model_version = "file_based"
            self.model_name = type(self.model).__name__
            print(f"Model loaded from file: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model from file: {str(e)}")
            return False

    def _fix_sklearn_compatibility(self):
        """Fixer les attributs manquants pour compatibilité sklearn"""
        try:
            from sklearn.linear_model import LogisticRegression

            if isinstance(self.model, LogisticRegression):
                # Ajouter les attributs manquants si nécessaire
                if not hasattr(self.model, 'multi_class'):
                    self.model.multi_class = 'auto'
                    print("Fixed: Added missing 'multi_class' attribute")

                if not hasattr(self.model, 'solver'):
                    self.model.solver = 'lbfgs'
                    print("Fixed: Added missing 'solver' attribute")

                if not hasattr(self.model, 'max_iter'):
                    self.model.max_iter = 100
                    print("Fixed: Added missing 'max_iter' attribute")

        except Exception as e:
            print(f"Warning: Could not fix sklearn compatibility: {e}")

    def _find_latest_model(self) -> Optional[str]:
        try:
            # Chercher le run_id dans run_metadata.json
            metadata_path = "run_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Le run_id devrait être dans les métadonnées MLflow
                    # Pour l'instant, on retourne None et on utilisera le fallback

            # Parcourir mlruns pour trouver un modèle
            mlruns_path = "mlruns"
            if os.path.exists(mlruns_path):
                # Chercher dans les artifacts des models/runs
                latest_model_path = None
                latest_time = 0

                for root, dirs, files in os.walk(mlruns_path):
                    # Chercher model.pkl dans les artifacts
                    if "model.pkl" in files:
                        model_pkl_path = os.path.join(root, "model.pkl")
                        # Vérifier la date de modification
                        mtime = os.path.getmtime(model_pkl_path)
                        if mtime > latest_time:
                            latest_time = mtime
                            # Utiliser le chemin du fichier pkl directement
                            latest_model_path = model_pkl_path

                if latest_model_path:
                    print(f"Found model at: {latest_model_path}")
                    return latest_model_path

            return None
        except Exception as e:
            print(f"Warning: Could not find latest model: {str(e)}")
            return None

    def _load_metadata(self):
        try:
            metadata_path = "run_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.n_features = metadata.get("n_features")
                    print(f"Model metadata loaded: {self.n_features} features")
        except Exception as e:
            print(f"Warning: Could not load metadata: {str(e)}")

    def _extract_feature_names(self):
        try:
            # Pour sklearn, essayer d'obtenir les feature_names
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                print(f"Feature names extracted: {len(self.feature_names)} features")
            else:
                print("Model does not have feature_names_in_ attribute")
                # Si pas de noms, on ne peut pas valider
                self.feature_names = None
        except Exception as e:
            print(f"Warning: Could not extract feature names: {str(e)}")
            self.feature_names = None

    def _prepare_features(self, features: Dict[str, float]) -> Dict[str, float]:
        if self.feature_names is None:
            # Si on n'a pas les noms de features, retourner tel quel
            return features

        # Créer un dict avec toutes les features attendues (valeur par défaut = 0)
        prepared = {name: 0.0 for name in self.feature_names}

        # Mettre à jour avec les features fournies
        for key, value in features.items():
            if key in prepared:
                prepared[key] = value
            else:
                print(f"Warning: Unknown feature '{key}' will be ignored")

        # Compter les features manquantes
        missing_features = [name for name in self.feature_names if name not in features]
        if missing_features:
            print(f"{len(missing_features)} features missing, filled with default value (0)")

        return prepared

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        try:
            # Préparer les features pour correspondre au modèle
            prepared_features = self._prepare_features(features)

            # Convertir en numpy array dans l'ordre des features attendues
            if self.feature_names:
                # Utiliser l'ordre des features du modèle
                feature_values = [prepared_features[name] for name in self.feature_names]
                X = np.array([feature_values])
            else:
                # Fallback: utiliser DataFrame
                df = pd.DataFrame([prepared_features])
                X = df.values

            # Prédiction avec gestion robuste des erreurs
            try:
                if hasattr(self.model, "predict_proba"):
                    # Classification avec probabilités
                    proba = self.model.predict_proba(X)
                    prediction_score = float(proba[0][1])  # Probabilité de la classe 1
                else:
                    # Régression ou autre
                    prediction_score = float(self.model.predict(X)[0])
            except AttributeError as ae:
                # Si erreur d'attribut, essayer avec DataFrame
                print(f"Warning: AttributeError with numpy array, trying DataFrame: {ae}")
                df = pd.DataFrame([prepared_features])
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(df)
                    prediction_score = float(proba[0][1])
                else:
                    prediction_score = float(self.model.predict(df)[0])

            return {
                "prediction_score": prediction_score,
                "n_features": len(prepared_features),
                "n_features_provided": len(features),
                "model_name": self.model_name
            }

        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "n_features": self.n_features,
            "feature_names": self.feature_names[:10] if self.feature_names else None,  # Premiers 10 noms
            "total_features": len(self.feature_names) if self.feature_names else None,
            "is_loaded": self.is_loaded()
        }

    def get_expected_features(self) -> Optional[list]:
        return self.feature_names

model_loader = ModelLoader()
