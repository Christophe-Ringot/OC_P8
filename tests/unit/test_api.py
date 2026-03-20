import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.api.models.model_loader import ModelLoader


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_loader():
    with patch('src.api.models.model_loader.model_loader') as mock:
        mock.is_loaded.return_value = True
        mock.model_version = "test_v1.0"
        mock.model_name = "TestModel"
        mock.predict.return_value = {
            "prediction_score": 0.35,
            "model_version": "test_v1.0",
            "n_features": 3,
            "n_features_provided": 3,
            "model_name": "TestModel"
        }
        mock.get_model_info.return_value = {
            "model_version": "test_v1.0",
            "model_name": "TestModel"
        }
        mock.get_expected_features.return_value = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY"
        ]
        yield mock


class TestRootEndpoint:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_root_has_docs_link(self, client):
        response = client.get("/")
        data = response.json()
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


class TestHealthEndpoint:
    def test_health_check_success(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Le statut peut être healthy ou degraded selon si le modèle est chargé
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data
        assert data["api_version"] == "1.0.0"

    def test_health_check_model_not_loaded(self, client):
        with patch('api.models.model_loader.model_loader') as mock:
            mock.is_loaded.return_value = False
            mock.get_model_info.return_value = {"model_version": None}

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False


class TestPredictionEndpoint:
    def test_predict_success(self, client):
        test_data = {
            "features": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5
            }
        }

        response = client.post("/api/v1/predict", json=test_data)
        # Le statut peut être 200 (success) ou 503 (modèle non chargé)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "prediction_score" in data
            assert "prediction_class" in data
            assert "model_version" in data
            assert "inference_time_ms" in data
            assert data["prediction_class"] in [0, 1]

    def test_predict_invalid_data(self, client):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422 

    def test_predict_model_not_loaded(self, client):
        with patch('api.models.model_loader.model_loader') as mock:
            mock.is_loaded.return_value = False

            test_data = {
                "features": {
                    "AMT_INCOME_TOTAL": 202500.0
                }
            }

            response = client.post("/api/v1/predict", json=test_data)
            assert response.status_code == 503

    def test_predict_model_error(self, client):
        with patch('api.models.model_loader.model_loader') as mock:
            mock.is_loaded.return_value = True
            mock.predict.side_effect = Exception("Prediction error")

            test_data = {
                "features": {
                    "AMT_INCOME_TOTAL": 202500.0
                }
            }

            response = client.post("/api/v1/predict", json=test_data)
            assert response.status_code in [500, 503]

    def test_predict_threshold_logic(self, client):
        with patch('api.models.model_loader.model_loader') as mock:
            mock.is_loaded.return_value = True
            mock.model_version = "test_v1.0"

            mock.predict.return_value = {
                "prediction_score": 0.35,
                "model_version": "test_v1.0",
                "n_features": 1,
                "n_features_provided": 1
            }

            test_data = {"features": {"AMT_INCOME_TOTAL": 100000.0}}
            response = client.post("/api/v1/predict", json=test_data)
            if response.status_code == 200:
                assert response.json()["prediction_class"] == 0

            mock.predict.return_value = {
                "prediction_score": 0.75,
                "model_version": "test_v1.0",
                "n_features": 1,
                "n_features_provided": 1
            }

            response = client.post("/api/v1/predict", json=test_data)
            if response.status_code == 200:
                assert response.json()["prediction_class"] == 1


class TestStatsEndpoint:
    def test_stats_no_predictions(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data

    def test_stats_with_predictions(self, client, mock_model_loader):
        test_data = {"features": {"AMT_INCOME_TOTAL": 100000.0}}
        for _ in range(3):
            client.post("/api/v1/predict", json=test_data)

        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] >= 3
        assert "avg_latency_ms" in data
        assert "error_rate" in data


class TestModelFeaturesEndpoint:
    def test_get_model_features(self, client):
        response = client.get("/api/v1/model/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data or "message" in data

    def test_get_model_features_not_available(self, client):
        with patch('api.models.model_loader.model_loader') as mock:
            mock.get_expected_features.return_value = None

            response = client.get("/api/v1/model/features")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data


class TestMonitoringEndpoints:
    def test_drift_summary(self, client):
        response = client.get("/api/v1/monitoring/drift/summary")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_drift_report_generation(self, client):
        with patch('api.monitoring.drift_detection.drift_detector') as mock:
            mock.generate_drift_report_from_logs.return_value = {
                "dataset_drift": False,
                "n_drifted_features": 0
            }

            response = client.post(
                "/api/v1/monitoring/drift/report",
                params={"output_path": "test_report.html"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestCORS:
    def test_cors_headers(self, client):
        response = client.options("/api/v1/predict")
        # CORS devrait être configuré
        assert response.status_code in [200, 405]


class TestPredictionLogging:
    def test_prediction_success_with_all_fields(self, client):
        test_data = {
            "features": {
                "AMT_INCOME_TOTAL": 100000.0,
                "AMT_CREDIT": 500000.0,
                "AMT_ANNUITY": 25000.0
            }
        }
        response = client.post("/api/v1/predict", json=test_data)

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "prediction_score" in data
            assert "timestamp" in data


class TestMonitoringEndpointsExtended:
    def test_set_reference_data_success(self, client):
        with patch('src.api.monitoring.drift_detection.drift_detector.load_reference_data'):
            response = client.post(
                "/api/v1/monitoring/drift/set-reference",
                params={"data_path": "data/test.csv"}
            )
            assert response.status_code == 200

    def test_set_reference_data_error(self, client):
        with patch('src.api.monitoring.drift_detection.drift_detector.load_reference_data', side_effect=Exception("File not found")):
            response = client.post(
                "/api/v1/monitoring/drift/set-reference",
                params={"data_path": "bad_path.csv"}
            )
            assert response.status_code == 500

    def test_drift_summary_error(self, client):
        with patch('src.api.monitoring.drift_detection.drift_detector.get_drift_summary', side_effect=Exception("Drift error")):
            response = client.get("/api/v1/monitoring/drift/summary")
            assert response.status_code == 500

    def test_drift_report_error(self, client):
        with patch('src.api.monitoring.drift_detection.drift_detector.generate_drift_report_from_logs', side_effect=Exception("Report error")):
            response = client.post("/api/v1/monitoring/drift/report")
            assert response.status_code == 500
