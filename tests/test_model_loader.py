import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.models.model_loader import ModelLoader


class TestModelLoader:
    @pytest.fixture
    def loader(self):
        return ModelLoader()

    @pytest.fixture
    def mock_sklearn_model(self):
        model = Mock()
        model.predict_proba = Mock(return_value=np.array([[0.65, 0.35]]))
        model.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
        return model

    def test_init(self, loader):
        assert loader.model is None
        assert loader.model_version is None
        assert loader.model_name is None
        assert loader.feature_names is None

    def test_is_loaded_false_initially(self, loader):
        assert loader.is_loaded() is False

    def test_is_loaded_true_after_loading(self, loader, mock_sklearn_model):
        loader.model = mock_sklearn_model
        assert loader.is_loaded() is True

    def test_load_model_from_file_success(self, loader, tmp_path):
        # Créer un mock de modèle
        mock_model = Mock()

        with patch('joblib.load', return_value=mock_model):
            result = loader.load_model_from_file("test_model.pkl")

            assert result is True
            assert loader.model is not None
            assert loader.model_version == "file_based"
            assert loader.model_name is not None

    def test_load_model_from_file_error(self, loader):
        result = loader.load_model_from_file("non_existent_model.pkl")
        assert result is False
        assert loader.model is None

    def test_extract_feature_names_with_sklearn(self, loader, mock_sklearn_model):
        loader.model = mock_sklearn_model
        loader._extract_feature_names()

        assert loader.feature_names == ['feature1', 'feature2', 'feature3']

    def test_extract_feature_names_without_attribute(self, loader):
        loader.model = Mock(spec=[])  # Mock sans feature_names_in_
        loader._extract_feature_names()

        assert loader.feature_names is None

    def test_prepare_features_all_provided(self, loader):
        loader.feature_names = ['feature1', 'feature2', 'feature3']

        input_features = {
            'feature1': 1.0,
            'feature2': 2.0,
            'feature3': 3.0
        }

        result = loader._prepare_features(input_features)

        assert result == input_features
        assert len(result) == 3

    def test_prepare_features_partial(self, loader):
        loader.feature_names = ['feature1', 'feature2', 'feature3']

        input_features = {
            'feature1': 1.0,
        }

        result = loader._prepare_features(input_features)

        assert result['feature1'] == 1.0
        assert result['feature2'] == 0.0  # Valeur par défaut
        assert result['feature3'] == 0.0  # Valeur par défaut
        assert len(result) == 3

    def test_prepare_features_with_unknown(self, loader):
        loader.feature_names = ['feature1', 'feature2']

        input_features = {
            'feature1': 1.0,
            'unknown_feature': 999.0  # Cette feature sera ignorée
        }

        result = loader._prepare_features(input_features)

        assert 'feature1' in result
        assert 'unknown_feature' not in result
        assert len(result) == 2

    def test_prepare_features_no_feature_names(self, loader):
        loader.feature_names = None

        input_features = {'feature1': 1.0, 'feature2': 2.0}
        result = loader._prepare_features(input_features)

        assert result == input_features

    def test_predict_success_with_proba(self, loader, mock_sklearn_model):
        loader.model = mock_sklearn_model
        loader.model_name = "TestModel"
        loader.feature_names = ['feature1', 'feature2', 'feature3']

        features = {
            'feature1': 1.0,
            'feature2': 2.0,
            'feature3': 3.0
        }

        result = loader.predict(features)

        assert 'prediction_score' in result
        assert 'n_features' in result
        assert 'n_features_provided' in result
        assert result['prediction_score'] == 0.35  # Classe 1
        assert result['n_features_provided'] == 3

    def test_predict_without_proba(self, loader):
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([42.5]))
        delattr(mock_model, 'predict_proba')  # Pas de predict_proba

        loader.model = mock_model
        loader.model_name = "RegressionModel"
        loader.feature_names = ['feature1']

        features = {'feature1': 1.0}
        result = loader.predict(features)

        assert result['prediction_score'] == 42.5

    def test_predict_model_not_loaded(self, loader):
        features = {'feature1': 1.0}

        with pytest.raises(ValueError, match="Model not loaded"):
            loader.predict(features)

    def test_predict_error_handling(self, loader):
        mock_model = Mock()
        mock_model.predict_proba = Mock(side_effect=Exception("Prediction failed"))

        loader.model = mock_model
        loader.model_name = "ErrorModel"
        loader.feature_names = ['feature1']

        features = {'feature1': 1.0}

        with pytest.raises(ValueError, match="Prediction error"):
            loader.predict(features)

    def test_get_model_info_loaded(self, loader, mock_sklearn_model):
        loader.model = mock_sklearn_model
        loader.model_name = "TestModel"
        loader.model_version = "v1.0"
        loader.feature_names = ['f1', 'f2', 'f3']

        info = loader.get_model_info()

        assert info['model_name'] == "TestModel"
        assert info['model_version'] == "v1.0"
        assert info['is_loaded'] is True
        assert info['total_features'] == 3

    def test_get_model_info_not_loaded(self, loader):
        info = loader.get_model_info()

        assert info['model_name'] is None
        assert info['model_version'] is None
        assert info['is_loaded'] is False

    def test_get_expected_features(self, loader):
        loader.feature_names = ['feature1', 'feature2']

        features = loader.get_expected_features()

        assert features == ['feature1', 'feature2']

    def test_get_expected_features_none(self, loader):
        assert loader.get_expected_features() is None

    def test_load_metadata_success(self, loader, tmp_path):
        # Créer un fichier de métadonnées temporaire
        metadata = {"n_features": 10, "model_type": "classifier"}

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                import json
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(metadata)

                loader._load_metadata()

                assert loader.n_features == 10

    def test_find_latest_model_not_found(self, loader):
        with patch('os.path.exists', return_value=False):
            result = loader._find_latest_model()
            assert result is None

    def test_load_model_from_mlflow_with_run_id(self, loader):
        with patch('mlflow.sklearn.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict_proba = Mock(return_value=np.array([[0.6, 0.4]]))
            mock_load.return_value = mock_model

            result = loader.load_model_from_mlflow(run_id="test_run_123")

            assert result is True
            assert loader.model_version == "test_run_123"
            mock_load.assert_called_once_with("runs:/test_run_123/model")

    def test_load_model_from_mlflow_error(self, loader):
        with patch('mlflow.sklearn.load_model', side_effect=Exception("MLflow error")):
            result = loader.load_model_from_mlflow(run_id="bad_run")
            assert result is False

    def test_fix_sklearn_compatibility(self, loader):
        from sklearn.linear_model import LogisticRegression

        # Créer un vrai modèle sans les attributs
        lr = LogisticRegression()
        loader.model = lr
        loader._fix_sklearn_compatibility()

        # Les attributs devraient exister maintenant
        assert hasattr(loader.model, 'multi_class')
        assert hasattr(loader.model, 'solver')
        assert hasattr(loader.model, 'max_iter')

    def test_predict_with_dataframe_fallback(self, loader):
        mock_model = Mock()
        # Simuler une erreur avec numpy array qui force le fallback
        mock_model.predict_proba = Mock(side_effect=[
            AttributeError("Array error"),
            np.array([[0.5, 0.5]])
        ])

        loader.model = mock_model
        loader.model_name = "TestModel"
        loader.feature_names = ['feature1']

        result = loader.predict({'feature1': 1.0})

        assert result['prediction_score'] == 0.5
        assert mock_model.predict_proba.call_count == 2

    def test_load_model_from_mlflow_without_run_id(self, loader):
        with patch.object(loader, '_find_latest_model', return_value=None):
            result = loader.load_model_from_mlflow()
            assert result is False

    def test_load_model_from_file_with_pkl(self, loader, tmp_path):
        import joblib

        # Créer un faux modèle
        mock_model = Mock()
        model_file = tmp_path / "model.pkl"

        with patch('joblib.load', return_value=mock_model):
            result = loader.load_model_from_file(str(model_file))
            assert result is True
            assert loader.model_version == "file_based"


class TestModelLoaderIntegration:
    def test_full_prediction_workflow(self):
        loader = ModelLoader()

        # Mock d'un modèle sklearn complet
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.7, 0.3]]))
        mock_model.feature_names_in_ = np.array(['amt_credit', 'amt_income'])

        loader.model = mock_model
        loader.model_name = "LogisticRegression"
        loader.model_version = "v1.0"
        loader._extract_feature_names()

        # Prédiction avec features partielles
        result = loader.predict({
            'amt_credit': 100000.0
            # amt_income manquant -> sera 0
        })

        assert result['prediction_score'] == 0.3
        assert result['n_features'] == 2
        assert result['n_features_provided'] == 1
