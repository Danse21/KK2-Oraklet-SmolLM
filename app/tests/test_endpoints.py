# Tests for all FastAPI endpoints using TestClient
# LLMRunner is mocked in the /ai/ask test thereby requiring no model download

from unittest.mock import patch
from app.schemas import ResponseParserOutput

def test_health_returns_ok(client):
  """GET /health returns 200 with {"status": "ok"}"""
  resp = client.get("/health")
  assert resp.status_code == 200
  assert resp.json() == {"status": "ok"}

