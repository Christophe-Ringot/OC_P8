import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.monitoring.drift_detection import DriftDetector


class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        return DriftDetector()

    @pytest.fixture
    def sample_reference_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 10, 1000),
            'feature2': np.random.normal(50, 5, 1000),
            'feature3': np.random.normal(200, 20, 1000)
        })

    @pytest.fixture
    def sample_production_data(self):
        np.random.seed(43)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 10, 500),
            'feature2': np.random.normal(50, 5, 500),
            'feature3': np.random.normal(200, 20, 500)
        })

    @pytest.fixture
    def drifted_production_data(self):
        np.random.seed(44)
        return pd.DataFrame({
            'feature1': np.random.normal(150, 15, 500),  # Drift significatif
            'feature2': np.random.normal(50, 5, 500),
            'feature3': np.random.normal(200, 20, 500)
        })

    def test_init_without_reference_data(self, detector):
        assert detector.reference_data is None
        assert detector.reference_data_path is None

    def test_init_with_reference_data_path(self, tmp_path, sample_reference_data):
        ref_file = tmp_path / "reference.csv"
        sample_reference_data.to_csv(ref_file, index=False)

        detector = DriftDetector(reference_data_path=str(ref_file))

        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000

    def test_load_reference_data_success(self, detector, tmp_path, sample_reference_data):
        ref_file = tmp_path / "reference.csv"
        sample_reference_data.to_csv(ref_file, index=False)

        detector.load_reference_data(str(ref_file))

        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000
        assert list(detector.reference_data.columns) == ['feature1', 'feature2', 'feature3']

    def test_load_reference_data_error(self, detector):
        detector.load_reference_data("non_existent.csv")
        assert detector.reference_data is None

    def test_prepare_production_data_from_logs_no_data(self, detector):
        with patch('sqlite3.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_connect.return_value.cursor.return_value = mock_cursor

            result = detector.prepare_production_data_from_logs("test.db")

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_compare_distributions_no_drift(self, detector, sample_reference_data, sample_production_data):
        indicators = detector._compare_distributions(sample_reference_data, sample_production_data)

        assert 'feature1' in indicators
        assert 'feature2' in indicators
        assert 'feature3' in indicators

        # Les moyennes devraient être similaires (pas de drift)
        assert indicators['feature1']['mean_drift_pct'] < 10
        assert indicators['feature2']['mean_drift_pct'] < 10
        assert indicators['feature3']['mean_drift_pct'] < 10

    def test_compare_distributions_with_drift(self, detector, sample_reference_data, drifted_production_data):
        indicators = detector._compare_distributions(sample_reference_data, drifted_production_data)

        # feature1 devrait avoir un drift détecté
        assert bool(indicators['feature1']['drift_detected']) is True
        assert indicators['feature1']['mean_drift_pct'] > 10

        # Les autres features ne devraient pas avoir de drift
        assert bool(indicators['feature2']['drift_detected']) is False
        assert bool(indicators['feature3']['drift_detected']) is False

    def test_get_drift_summary_no_data(self, detector):
        summary = detector.get_drift_summary("test.db")

        assert summary['status'] == 'no_data'
        assert 'message' in summary

    def test_get_drift_summary_no_reference(self, detector):
        with patch.object(detector, 'prepare_production_data_from_logs') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame({'feature1': [1, 2, 3]})

            summary = detector.get_drift_summary("test.db")

            assert summary['status'] == 'no_data'

    def test_get_drift_summary_success(self, detector, sample_reference_data, sample_production_data):
        detector.reference_data = sample_reference_data

        # Ajouter une colonne timestamp
        sample_production_data['timestamp'] = pd.date_range('2024-01-01', periods=len(sample_production_data))

        with patch.object(detector, 'prepare_production_data_from_logs') as mock_prepare:
            mock_prepare.return_value = sample_production_data

            summary = detector.get_drift_summary("test.db")

            assert summary['status'] == 'ok'
            assert summary['n_reference_samples'] == 1000
            assert summary['n_production_samples'] == 500
            assert summary['n_common_features'] == 3
            assert 'drift_indicators' in summary

    def test_detect_drift_no_reference_data(self, detector, sample_production_data):
        # Si Evidently n'est pas disponible, ça retourne un dict avec error
        result = detector.detect_drift(sample_production_data)
        # Soit une ValueError, soit un dict avec erreur
        if isinstance(result, dict):
            assert 'error' in result or result is None
        else:
            pytest.fail("Should handle missing reference data")

    def test_detect_drift_empty_production_data(self, detector, sample_reference_data):
        detector.reference_data = sample_reference_data

        # Si Evidently n'est pas disponible, ça retourne un dict avec error
        result = detector.detect_drift(pd.DataFrame())
        if isinstance(result, dict):
            assert 'error' in result
        else:
            pytest.fail("Should handle empty production data")

    def test_detect_drift_no_common_columns(self, detector):
        detector.reference_data = pd.DataFrame({'col1': [1, 2, 3]})
        production_data = pd.DataFrame({'col2': [4, 5, 6]})

        # Si Evidently n'est pas disponible, ça retourne un dict avec error
        result = detector.detect_drift(production_data)
        if isinstance(result, dict):
            assert 'error' in result
        else:
            pytest.fail("Should handle no common columns")

    @patch('api.monitoring.drift_detection.EVIDENTLY_AVAILABLE', True)
    @patch('api.monitoring.drift_detection.DataDriftPreset')
    @patch('api.monitoring.drift_detection.DataQualityPreset')
    @patch('api.monitoring.drift_detection.Report')
    def test_detect_drift_success(self, MockReport, MockDataQuality, MockDataDrift, detector, sample_reference_data, sample_production_data, tmp_path):
        detector.reference_data = sample_reference_data

        mock_report = MagicMock()
        mock_report.as_dict.return_value = {
            'metrics': [
                {
                    'metric': 'DatasetDriftMetric',
                    'result': {
                        'dataset_drift': False,
                        'drift_by_columns': {
                            'feature1': False,
                            'feature2': False,
                            'feature3': False
                        }
                    }
                }
            ]
        }
        MockReport.return_value = mock_report

        output_path = str(tmp_path / "drift_report.html")
        result = detector.detect_drift(sample_production_data, output_path)

        assert 'dataset_drift' in result
        assert result['dataset_drift'] is False
        assert result['n_drifted_features'] == 0

    @patch('api.monitoring.drift_detection.EVIDENTLY_AVAILABLE', False)
    def test_detect_drift_evidently_not_available(self, detector, sample_reference_data, sample_production_data):
        detector.reference_data = sample_reference_data

        result = detector.detect_drift(sample_production_data)

        assert 'error' in result
        assert 'Evidently not available' in result['error']

    def test_extract_drift_metrics_with_drift(self, detector):
        report_dict = {
            'metrics': [
                {
                    'metric': 'DatasetDriftMetric',
                    'result': {
                        'dataset_drift': True,
                        'drift_by_columns': {
                            'feature1': True,
                            'feature2': False,
                            'feature3': True
                        }
                    }
                }
            ]
        }

        result = detector._extract_drift_metrics(report_dict)

        assert result['dataset_drift'] is True
        assert result['n_drifted_features'] == 2
        assert 'feature1' in result['drifted_features']
        assert 'feature3' in result['drifted_features']
        assert 'timestamp' in result

    def test_extract_drift_metrics_error_handling(self, detector):
        # Dict mal formé
        report_dict = {'invalid': 'structure'}

        result = detector._extract_drift_metrics(report_dict)

        # Devrait retourner un dict avec dataset_drift à None
        assert result['dataset_drift'] is None

    def test_generate_drift_report_from_logs_no_production_data(self, detector):
        with patch.object(detector, 'prepare_production_data_from_logs') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame()

            result = detector.generate_drift_report_from_logs("test.db")

            assert 'error' in result
            assert result['dataset_drift'] is None

    def test_compare_distributions_with_timestamp(self, detector, sample_reference_data):
        ref_data = sample_reference_data.copy()
        ref_data['timestamp'] = pd.date_range('2024-01-01', periods=len(ref_data))

        prod_data = sample_reference_data.copy()
        prod_data['timestamp'] = pd.date_range('2024-02-01', periods=len(prod_data))

        indicators = detector._compare_distributions(ref_data, prod_data)

        # timestamp ne devrait pas être dans les indicateurs
        assert 'timestamp' not in indicators
        assert 'feature1' in indicators

    def test_compare_distributions_zero_reference_mean(self, detector):
        ref_data = pd.DataFrame({'feature1': [0, 0, 0]})
        prod_data = pd.DataFrame({'feature1': [1, 2, 3]})

        indicators = detector._compare_distributions(ref_data, prod_data)

        assert indicators['feature1']['mean_drift_pct'] == 0


class TestDriftDetectorIntegration:
    def test_full_drift_detection_workflow(self, tmp_path):
        # Créer des données de référence
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'amt_credit': np.random.normal(100000, 10000, 1000),
            'amt_income': np.random.normal(50000, 5000, 1000)
        })

        ref_file = tmp_path / "reference.csv"
        reference_data.to_csv(ref_file, index=False)

        # Initialiser le detector
        detector = DriftDetector(reference_data_path=str(ref_file))

        # Créer des données de production avec drift
        np.random.seed(43)
        production_data = pd.DataFrame({
            'amt_credit': np.random.normal(150000, 15000, 500),  # Drift
            'amt_income': np.random.normal(50000, 5000, 500)
        })
        production_data['timestamp'] = pd.date_range('2024-01-01', periods=len(production_data))

        # Mock de prepare_production_data_from_logs
        with patch.object(detector, 'prepare_production_data_from_logs') as mock_prepare:
            mock_prepare.return_value = production_data

            # Obtenir le résumé
            summary = detector.get_drift_summary("test.db")

            assert summary['status'] == 'ok'
            assert bool(summary['drift_indicators']['amt_credit']['drift_detected']) is True
            assert bool(summary['drift_indicators']['amt_income']['drift_detected']) is False
