# Handles everything related to the uploaded CSV dataset

import io
import pandas as pd
import numpy as np
from fastapi import HTTPException

_dataset: pd.DataFrame | None = None

def get_dataset() -> pd.DataFrame:
  # Raise 404 if not dataset is uploaded yet
  if _dataset is None:
    raise HTTPException(
      status_code=404,
      detail="No dataset found. Check if file was correctly uploaded."
    )
  return _dataset

# Reads a CSV file from raw bytes, validates it, and stores it in memory
def load_csv(file_bytes: bytes, filename: str) -> dict:
  global _dataset

  # check the file extension matches
  if not filename.lower().endswith(".csv"):
    raise HTTPException(
      status_code=400,
      detail=f"Invalid file type: '{filename}'. Only .csv files are acceptable."
    )
  # Try to read the file with Pandas
  try:
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", encoding_error="strict")
  # Raise an error if file is not a valid CSV
  except Exception as e:
    raise HTTPException(
      status_code=400,
      detail=f"The file could not be read as CSV: {e}"
    )

  # Check that the file is not empty
  if df.empty:
    raise HTTPException(
      status_code=400,
      detail="The file is empty. Upload a CSV file with data."
    )

  # Rename columns to match that of KK1 notebook (analyzed data)
  rename_map = {
        "Overall rank":               "rank",
        "Country or region":          "country",
        "Score":                      "happiness_score",
        "GDP per capita":             "gdp_per_capita",
        "Social support":             "social_support",
        "Healthy life expectancy":    "life_expectancy",
        "Freedom to make life choices": "freedom",
        "Generosity":                 "generosity",
        "Perceptions of corruption":  "corruption",
    }

  # apply renaming to only columns that exist in the file
  df =df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

  # store the cleaned dataset in memory
  _dataset = df

  # Build metadata response
  return {
    "rows": len(df),
    "columns": df.columns.tolist(),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
  }

# Returns descriptive statistics from the stored dataset
def get_stats() -> dict:
  df = get_dataset()

  # Transposes the returned statistics so each column becomes a key in the dict
  # Remove rank and country in statistics estimation
  stats = df.drop(columns=["rank", "country"], errors='ignore').describe().to_dict()

  return {"stats": stats}

# Returns the number of rows and columns, just like df.shape
# Get called by GET /data/shape
def get_shape() -> dict:
  df = get_dataset()

  # df.shape returns a tuple (rows, columns)
  rows, columns = df.shape
  return {"rows": rows, "columns": columns}

# Returns the top or bottom n countries ranked by happiness score
# mirrors df.nlargest() and df.nsmallest()
# Called by GET /data/top and GET /data/bottom
def get_top_bottom(n: int, order: str) -> dict:
  df = get_dataset()

  # Check that happiness_score column exist
  if "happiness_score" not in df.columns:
    raise HTTPException(
      status_code=400,
      detail="Dataset has no 'happiness_score' column"
    )

  # Check that country column exist
  if "country" not in df.columns:
    raise HTTPException(
      status_code=400,
      detail="Dataset has no 'country' column."
    )

  # Select the n largest or smallest rows by score
  if order == "top":
    selected = df.nlargest(n, "happiness_score")[["country", "happiness_score"]]
  else:
    selected = df.nsmallest(n, "happiness_score")[["country", "happiness_score"]]

  # Build the result as a list of dicts
  results = [
    {"country": row["country"], "happiness_score": round(row["happiness_score"], 3)}
    for _, row in selected.iterrows()
  ]
  return {"results": results}

# Finds countries with a value of 0.000
# Called by GET /data/zeros
def get_zeros() -> dict:
  df = get_dataset()
  zeros = {}

  # Iterate over columns with numeric values only
  for col in df.select_dtypes(include="number").columns:
    zero_rows = df[df[col] == 0.000]  # Finds rows where column equals 0.000

    # Build result for columns that contain zeros only
    if not zero_rows.empty:
      if "country" in df.columns:
        zeros[col] = zero_rows["country"].tolist()  # Gives country name for those rows
      else:
        zeros[col] = zero_rows.index.tolist() # No country column?, return the row indices
  return {"zeros": zeros}

# Counts NaN (ie., null) values per column in the dataset
# Called by GET /data/missing
def get_missing() -> dict:
  df = get_dataset()

  missing_counts = df.isnull().sum()

  # Filter columns with at least one NaN
  missing = {
    col: int(count)
    for col, count in missing_counts.items()
    if count > 0
  }
  return {"missing": missing}

# Returns countries that over or under performed their
# would be predicted happiness score based on their GDP per capita
# Caled by GET /data/outliers
def get_outliers(n: int = 5) -> dict:
  df = get_dataset()

  # Validate that both columns exist in the dataset
  for col in ["gdp_per_capita", "happiness_score"]:
    if col not in df.columns:
      raise HTTPException(
        status_code=400,
        detail=f"No '{col}' in the dataset."
      )
  # check that country exist
  if "country" not in df.columns:
    raise HTTPException(
      status_code=400,
      detail="No 'country' in the dataset"
    )

  # Drop any row with NaN
  clean = df[["gdp_per_capita", "happiness_score", "country"]].dropna()

  # Fit a degree 1 polynomial through gdp per capita and happiness score ie., a straight line
  coeffs = np.polyfit(clean["gdp_per_capita"], clean["happiness_score"], deg=1)

  # Predict the happiness score for each country based on their GDP value only
  clean = clean.copy()
  clean["predicted"] = np.polyval(coeffs, clean["gdp_per_capita"])
  clean["residual"] = clean["happiness_score"] - clean["predicted"]
  over_performers = clean.nlargest(n, "residual")[["country", "happiness_score"]]
  under_performers = clean.nsmallest(n, "residual")[["country", "happiness_score"]]

  over_performers_list = [
    {"country": row["country"], "happiness_score": round(row["happiness_score"], 3)}
    for _, row in over_performers.iterrows()
    ]
  under_performers_list = [
    {"country": row["country"], "happiness_score": round(row["happiness_score"], 3)}
    for _, row in under_performers.iterrows()
  ]
  return {"over_performers": over_performers_list, "under_performers": under_performers_list}


# Estimate and return correlation with happiness score
def get_correlations() -> (dict):
  df = get_dataset()

  expected_cols = ["gdp_per_capita", "social_support", "life_expectancy", "freedom", "generosity", "corruption"]
  numeric_cols = [col for col in expected_cols if col in df.columns]

  correlations = (df[numeric_cols]
                  .corrwith(df["happiness_score"])
                  .sort_values(ascending=False)
                  .round(3)
                  .to_dict()
  )
  return {"correlations": correlations}
