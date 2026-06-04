# Contains shares fixtures for all tests in both test_chain.py and test_endpoints.py
# pytest automatically loads this file before running any test

import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.data as data_module

# And run automatically (autouse=True) for every test
@pytest.fixture(autouse=True)
def reset_dataset():
  """Resets the in-memory dataset to None before and after every test"""
  data_module._dataset = None
  yield
  data_module._dataset = None

# Returns a FastAPI TestClient connected to the app.
# TestClient makes it possible to make HTTP requests to the app in tests 
# without actually starting a server
@pytest.fixture
def client() -> TestClient:
  return TestClient(app)

@pytest.fixture
def sample_csv_bytes() -> bytes:
  """Small valid CSV with 3 rows of World Happiness data for upload tests."""
  csv_content = (
    "Overall rank,Country or region,Score,GDP per capita,"
    "Social support,Healthy life expectancy,"
    "Freedom to make life choices,Generosity,Perceptions of corruption\n"
    "1,Finland,7.769,1.340,1.587,0.986,0.596,0.153,0.393\n"
    "2,Denmark,7.600,1.383,1.573,0.996,0.592,0.252,0.410\n"
    "3,Norway,7.554,1.488,1.582,1.028,0.603,0.271,0.341\n"
  )
  return csv_content.encode("utf-8")

@pytest.fixture
def sample_csv_bytes_with_zero() -> bytes:
  """CSV with Somalia (gdp=0.0) for testing GET /data/zeros."""
  csv_content = (
    "Overall rank,Country or region,Score,GDP per capita,"
    "Social support,Healthy life expectancy,"
    "Freedom to make life choices,Generosity,Perceptions of corruption\n"
    "1,Finland,7.769,1.340,1.587,0.986,0.596,0.153,0.393\n"
    "2,Somalia,2.900,0.000,0.698,0.268,0.559,0.243,0.270\n"
  )
  return csv_content.encode("utf-8")