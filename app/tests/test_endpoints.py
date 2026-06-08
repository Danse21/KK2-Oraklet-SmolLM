# Tests for all FastAPI endpoints using TestClient
# LLMRunner is mocked in the /ai/ask test thereby requiring no model download

# Run with: uv run pytest app/tests/test_endpoints.py -v

from unittest.mock import patch
from app.schemas import ResponseParserOutput
from app.chain.runnable import RunnableSequence

def test_health_returns_ok(client):
  """GET /health returns 200 with {"status": "ok"}"""
  resp = client.get("/health")
  assert resp.status_code == 200
  assert resp.json() == {"status": "ok"}

def test_upload_valid_csv_returns_metadata(client, sample_csv_bytes):
  """POST /data/upload with a valid CSV returns 200 with rows, columns and dtypes"""
  resp = client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  assert resp.status_code == 200
  data = resp.json()
  assert "rows" in data
  assert "columns" in data
  assert "dtypes" in data
  assert data["rows"] == 3

def test_upload_wrong_file_extension_returns_400(client):
  """POST /data/upload with a .txt file returns 400."""
  resp = client.post(
    "/data/upload",
    files={"file": ("report.txt", b"this is not a csv file", "text/plain")}
  )
  assert resp.status_code == 400

def test_upload_empty_file_returns_400(client):
  """POST /data/upload with an empty file returns 400."""
  resp = client.post(
    "/data/upload",
    files={"file": ("empty.csv", b"", "text/csv")}
  )
  assert resp.status_code == 400

def test_stats_without_upload_returns_400(client):
  """GET /data/stats without an uploaded dataset returns 404."""
  resp = client.get("/data/stats")
  assert resp.status_code == 404

def test_stats_after_upload_returns_200(client, sample_csv_bytes):
  """GET /data/stats after upload returns 200 with statistics."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/stats")
  assert resp.status_code == 200
  data = resp.json()
  assert "stats" in data
  assert len(data["stats"]) > 0

def test_correlations_without_upload_returns_404(client):
  """GET /data/correlations without an uploaded dataset returns 404."""
  resp = client.get("/data/correlations")
  assert resp.status_code == 404

def test_correlations_after_upload_returns_200(client, sample_csv_bytes):
  """GET /data/correlations after upload returns 200 with correlation values."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/correlations")
  assert resp.status_code == 200
  data = resp.json()
  assert "correlations" in data
  assert len(data["correlations"]) > 0
  for col, value in data["correlations"].items():
    assert -1.0 <= value <= 1.0, f"Correlation for '{col}' is {value}, outside [-1, 1]."

def test_shape_without_upload_returns_404(client):
  """GET /data/shape without an uploaded dataset returns 404."""
  resp = client.get("/data/shape")
  assert resp.status_code == 404

def test_shape_after_upload_returns_correct_dimensions(client, sample_csv_bytes):
  """GET /data/shape returns the correct number of rows and columns."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/shape")
  assert resp.status_code == 200
  data = resp.json()
  assert "rows" in data
  assert "columns" in data
  assert data["rows"] == 3
  assert data["columns"] == 9

def test_top_without_upload_returns_404(client):
  """GET /data/top without an uploaded dataset returns 404."""
  resp = client.get("/data/top")
  assert resp.status_code == 404

def test_top_with_unrecognized_columns_returns_400(client):
  """GET /data/top after uploading a CSV file with unrecognized columns returns 400."""
  unmatched_csv = b"foo,bar,baz\n1,2,3\n4,5,6\n"
  client.post("/data/upload", files={"file": ("unmatched.csv", unmatched_csv, "text/csv")},)
  resp = client.get("/data/top")
  assert resp.status_code == 400

def test_top_returns_correct_number_of_countries(client, sample_csv_bytes):
  """GET /data/top?n=2 returns 2 countries with Finland first."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/top?n=2")
  assert resp.status_code == 200
  data = resp.json()
  assert "results" in data
  assert len(data["results"]) == 2
  assert data["results"][0]["country"] == "Finland"

def test_top_default_n_does_not_exceed_dataset_size(client, sample_csv_bytes):
  """GET /data/top with n=10 on a 3-row dataset returns all 3 rows."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/top")
  assert resp.status_code == 200
  assert len(resp.json()["results"]) == 3

def test_bottom_without_upload_returns_404(client):
  """GET /data/bottom without an uploaded dataset returns 404."""
  resp = client.get("/data/bottom")
  assert resp.status_code == 404

def test_bottom_returns_lowest_scoring_country_first(client, sample_csv_bytes):
  """GET /data/bottom?n=2 returns 2 countries with Norway (lowest) first."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/bottom?n=2")
  assert resp.status_code == 200
  data = resp.json()
  assert "results" in data
  assert len(data["results"]) == 2
  assert data["results"][0]["country"] == "Norway"

def test_zeros_without_upload_returns_404(client):
  """GET /data/zeros without an uploaded dataset returns 404."""
  resp = client.get("/data/zeros")
  assert resp.status_code == 404

def test_zeros_returns_empty_when_no_zeros_present(client, sample_csv_bytes):
  """GET /data/zeros on a clean CSV with no zero values returns an empty zeros dict."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/zeros")
  assert resp.status_code == 200
  assert resp.json()["zeros"] == {}


def test_zeros_detects_zero_values(client, sample_csv_bytes_with_zero):
  """GET /data/zeros detects Somalia with gdp=0.0 and reports it in the zeros dict."""
  client.post(
    "/data/upload",
    files={"file": ("zeros_test.csv", sample_csv_bytes_with_zero, "text/csv")},
  )
  resp = client.get("/data/zeros")
  assert resp.status_code == 200
  data = resp.json()
  assert "gdp_per_capita" in data["zeros"]
  assert "Somalia" in data["zeros"]["gdp_per_capita"]

def test_missing_without_upload_returns_404(client):
  """GET /data/missing without an uploaded dataset returns 404."""
  resp = client.get("/data/missing")
  assert resp.status_code == 404


def test_missing_returns_empty_for_complete_dataset(client, sample_csv_bytes):
  """GET /data/missing on a complete CSV returns an empty missing dict."""
  client.post(
      "/data/upload",
      files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/missing")
  assert resp.status_code == 200
  assert resp.json()["missing"] == {}

def test_outliers_without_upload_returns_404(client):
  """GET /data/outliers without an uploaded dataset returns 404."""
  resp = client.get("/data/outliers")
  assert resp.status_code == 404


def test_outliers_returns_correct_structure(client, sample_csv_bytes):
  """GET /data/outliers returns over_performers and under_performers with country and happiness_score."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  resp = client.get("/data/outliers?n=1")
  assert resp.status_code == 200
  data = resp.json()
  assert "over_performers" in data
  assert "under_performers" in data
  for entry in data["over_performers"] + data["under_performers"]:
    assert "country" in entry
    assert "happiness_score" in entry

def test_ask_without_upload_returns_404(client):
  """POST /ai/ask without an uploaded dataset returns 404."""
  resp = client.post("/ai/ask", json={"question": "What is the mean score?"})
  assert resp.status_code == 404


def test_ask_with_mocked_model_returns_answer(client, sample_csv_bytes):
  """POST /ai/ask with a mocked LLMRunner returns 200 with question, answer and model."""
  client.post(
    "/data/upload",
    files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")},
  )
  fake_result = ResponseParserOutput(
    question="What is the mean score?",
    answer="The mean happiness score is 5.4.",
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
  )
  with patch.object(RunnableSequence, 'invoke', return_value=fake_result):
    resp = client.post(
      "/ai/ask",
      json={"question": "What is the mean score?"},
    )
  assert resp.status_code == 200
  data = resp.json()
  assert "question" in data
  assert "answer" in data
  assert "model" in data
  assert data["answer"] == "The mean happiness score is 5.4."

def test_ask_returns_500_when_chain_raises(client, sample_csv_bytes):
  """POST /ai/ask returns 500 when the chain raises an unexpected exception."""
  client.post("/data/upload", files={"file": ("data-2019.csv", sample_csv_bytes, "text/csv")})
  with patch.object(RunnableSequence, 'invoke', side_effect=RuntimeError("model timeout")):
    resp = client.post("/ai/ask", json={"question": "What is the mean score?"},)
    assert resp.status_code == 500
