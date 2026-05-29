# Handles everything related to the uploaded CSV dataset

import io
import pandas as pd
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
def load_csv(file_size: bytes, filename: str) -> dict:
  global _dataset

  # check the file extension matches
  if not filename.lower().endswith(".csv"):
    raise HTTPException(
      status_code=400,
      detail=f"Invalid file type: '{filename}'. Only .csv files are acceptable."
    )
  # Try to read the file with Pandas
  try:
    df = pd.read_csv(io.BytesIO(file_size))
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
        "Country":                    "country",
        "Score":                      "score",
        "GDP per capita":             "gdp",
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
  stats = df.describe().to_dict()

  return {"stats": stats}