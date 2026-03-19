import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_db_path():
    return "test_logs.db"


@pytest.fixture(autouse=True)
def cleanup_test_db(test_db_path):
    yield
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass


@pytest.fixture
def sample_features():
    return {
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637
    }
