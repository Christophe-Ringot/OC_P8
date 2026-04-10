import pandas as pd
import os
from datetime import datetime
import json

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        ColumnDriftMetric,
        DatasetMissingValuesMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Evidently not available: {str(e)}")
    EVIDENTLY_AVAILABLE = False
    Report = None
    DataDriftPreset = None
    DataQualityPreset = None


class DriftDetector:

    def __init__(self, reference_data_path: str = None):
        self.reference_data = None
        self.reference_data_path = reference_data_path

        if reference_data_path and os.path.exists(reference_data_path):
            self.load_reference_data(reference_data_path)

    def load_reference_data(self, path: str):
        try:
            self.reference_data = pd.read_csv(path)
            print(f"Reference data loaded: {len(self.reference_data)} rows")
        except Exception as e:
            print(f"Error loading reference data: {str(e)}")

    def prepare_production_data_from_logs(self, db_path: str = "logs.db") -> pd.DataFrame:
        import sqlite3

        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT
                    input_features,
                    prediction_score,
                    timestamp
                FROM prediction_logs
                WHERE status_code = 200
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("No production data available")
                return pd.DataFrame()

            # Convertir les JSON features en colonnes
            features_df = pd.json_normalize(df['input_features'].apply(json.loads))
            features_df['prediction_score'] = df['prediction_score'].values
            features_df['timestamp'] = pd.to_datetime(df['timestamp'])

            print(f"Production data prepared: {len(features_df)} rows")
            return features_df

        except Exception as e:
            print(f"Error preparing production data: {str(e)}")
            return pd.DataFrame()

    def detect_drift(
        self,
        production_data: pd.DataFrame,
        output_path: str = "reports/drift_report.html"
    ) -> dict:
        if not EVIDENTLY_AVAILABLE:
            return {
                "error": "Evidently not available",
                "message": "Please install evidently: pip install evidently==0.4.33 --upgrade"
            }

        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        if production_data.empty:
            raise ValueError("Production data is empty")

        try:
            # Aligner les colonnes (garder seulement celles en commun)
            common_columns = list(
                set(self.reference_data.columns) & set(production_data.columns)
            )

            if not common_columns:
                raise ValueError("No common columns between reference and production data")

            ref_data = self.reference_data[common_columns]
            prod_data = production_data[common_columns]

            # Créer le rapport Evidently
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])

            report.run(
                reference_data=ref_data,
                current_data=prod_data
            )

            # Sauvegarder le rapport HTML
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            report.save_html(output_path)
            print(f"Drift report saved to: {output_path}")

            # Extraire les métriques principales
            report_dict = report.as_dict()
            drift_results = self._extract_drift_metrics(report_dict)

            return drift_results

        except Exception as e:
            print(f"Error detecting drift: {str(e)}")
            raise

    def _extract_drift_metrics(self, report_dict: dict) -> dict:
        try:
            metrics = report_dict.get("metrics", [])

            # Chercher la métrique de drift du dataset
            dataset_drift = None
            drifted_features = []
            drift_details = {}

            for metric in metrics:
                metric_name = metric.get("metric")

                if metric_name == "DatasetDriftMetric":
                    result = metric.get("result", {})
                    dataset_drift = result.get("dataset_drift", False)
                    drift_by_columns = result.get("drift_by_columns", {})

                    for col, is_drifted in drift_by_columns.items():
                        if is_drifted:
                            drifted_features.append(col)
                            drift_details[col] = {"drifted": True}

            return {
                "dataset_drift": dataset_drift,
                "n_drifted_features": len(drifted_features),
                "drifted_features": drifted_features,
                "drift_details": drift_details,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f"Warning: Could not extract drift metrics: {str(e)}")
            return {
                "dataset_drift": None,
                "error": str(e)
            }

    def generate_drift_report_from_logs(
        self,
        db_path: str = "logs.db",
        output_path: str = "reports/drift_report.html"
    ) -> dict:
        # Préparer les données de production
        production_data = self.prepare_production_data_from_logs(db_path)

        if production_data.empty:
            return {
                "error": "No production data available",
                "dataset_drift": None
            }

        # Détecter le drift
        return self.detect_drift(production_data, output_path)

    def get_drift_summary(self, db_path: str = "logs.db") -> dict:
        try:
            production_data = self.prepare_production_data_from_logs(db_path)

            if production_data.empty or self.reference_data is None:
                return {
                    "status": "no_data",
                    "message": "Insufficient data for drift detection"
                }

            # Aligner les colonnes
            common_columns = list(
                set(self.reference_data.columns) & set(production_data.columns)
            )

            if not common_columns:
                return {
                    "status": "no_common_columns",
                    "message": "No common columns between reference and production"
                }

            ref_data = self.reference_data[common_columns]
            prod_data = production_data[common_columns]

            # Comparaison des distributions
            drift_indicators = self._compare_distributions(ref_data, prod_data)

            # Statistiques de base
            return {
                "status": "ok",
                "n_reference_samples": len(ref_data),
                "n_production_samples": len(prod_data),
                "n_common_features": len(common_columns),
                "production_date_range": {
                    "start": production_data['timestamp'].min().isoformat() if 'timestamp' in production_data else None,
                    "end": production_data['timestamp'].max().isoformat() if 'timestamp' in production_data else None
                },
                "drift_indicators": drift_indicators
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _compare_distributions(self, ref_data: pd.DataFrame, prod_data: pd.DataFrame) -> dict:
        """Compare les distributions entre référence et production"""
        indicators = {}

        for col in ref_data.columns:
            if col == 'timestamp':
                continue

            try:
                ref_mean = ref_data[col].mean()
                prod_mean = prod_data[col].mean()
                ref_std = ref_data[col].std()
                prod_std = prod_data[col].std()

                # Calculer l'écart relatif des moyennes
                if ref_mean != 0:
                    mean_drift_pct = abs((prod_mean - ref_mean) / ref_mean) * 100
                else:
                    mean_drift_pct = 0

                # Alerte si écart > 10%
                drift_detected = mean_drift_pct > 10

                indicators[col] = {
                    "ref_mean": float(ref_mean),
                    "prod_mean": float(prod_mean),
                    "ref_std": float(ref_std),
                    "prod_std": float(prod_std),
                    "mean_drift_pct": float(mean_drift_pct),
                    "drift_detected": bool(drift_detected)  # Convertir numpy.bool en bool Python
                }
            except Exception as e:
                print(f"Error comparing column {col}: {str(e)}")
                continue

        return indicators

# Initialiser avec les données de référence si disponibles
_reference_paths = [
    "data/reference/reference_data.csv",
    "data/dataset_final.csv"  # Fallback sur le dataset complet
]

_reference_path = None
for path in _reference_paths:
    if os.path.exists(path):
        _reference_path = path
        break

drift_detector = DriftDetector(reference_data_path=_reference_path)
