"""
Script de test de l'API
"""
import pytest
import requests
import json
import time

API_URL = "http://localhost:8000"


@pytest.mark.integration
def test_health_check():
    """Test du endpoint health"""
    print("\n🔍 Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


@pytest.mark.integration
def test_prediction():
    """Test du endpoint de prédiction"""
    print("\n🔍 Testing prediction...")

    # Données de test (exemple de features)
    test_data = {
        "features": {
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5,
            "DAYS_BIRTH": -9461,
            "DAYS_EMPLOYED": -637,
            "DAYS_ID_PUBLISH": -2120,
            "AMT_GOODS_PRICE": 351000.0,
            "REGION_POPULATION_RELATIVE": 0.0184,
            "DAYS_REGISTRATION": -3648.0,
            "HOUR_APPR_PROCESS_START": 10
        }
    }

    response = requests.post(
        f"{API_URL}/api/v1/predict",
        json=test_data
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        print(f"\n✅ Prediction: {result['prediction_class']} (score: {result['prediction_score']:.4f})")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False


@pytest.mark.integration
def test_stats():
    """Test du endpoint de stats"""
    print("\n🔍 Testing stats...")
    response = requests.get(f"{API_URL}/api/v1/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


@pytest.mark.integration
def test_drift_summary():
    """Test du endpoint de drift summary"""
    print("\n🔍 Testing drift summary...")
    response = requests.get(f"{API_URL}/api/v1/monitoring/drift/summary")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def run_multiple_predictions(n=10):
    """Lance plusieurs prédictions pour tester le monitoring"""
    print(f"\n🔄 Running {n} predictions...")

    test_features = [
        {
            "AMT_INCOME_TOTAL": 202500.0 + i * 10000,
            "AMT_CREDIT": 406597.5 + i * 5000,
            "AMT_ANNUITY": 24700.5 + i * 1000,
            "DAYS_BIRTH": -9461 - i * 100,
            "DAYS_EMPLOYED": -637 - i * 10,
        }
        for i in range(n)
    ]

    success_count = 0
    for i, features in enumerate(test_features):
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            success_count += 1
            print(f"✅ Prediction {i+1}/{n} success")
        else:
            print(f"❌ Prediction {i+1}/{n} failed")
        time.sleep(0.1)

    print(f"\n📊 Results: {success_count}/{n} successful predictions")
    return success_count == n


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Credit Risk API")
    print("=" * 60)

    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed - is the API running?")
        print("💡 Start the API with: python start_api.py")
        exit(1)

    # Test prediction
    test_prediction()

    # Run multiple predictions
    run_multiple_predictions(5)

    # Test stats
    test_stats()

    # Test drift summary
    test_drift_summary()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n📊 Check the dashboard at: http://localhost:8501")
    print("📚 Check the API docs at: http://localhost:8000/docs")
