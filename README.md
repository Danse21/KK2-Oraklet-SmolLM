# KK2-Oraklet-SmolLM project:

A system built with a **REST API in FastAPI** that enables users to upload datasets(.csv file), perform automated data analysis and ask natural natural language questions about their data.

## Key features

- **Structured LLM chain:** Implements a predictable, typed pipeline (PromptBuilder | LLMRunner | ResponseParser) backed by hosted **Groq API**.
- **Automated Data analysis:** Integrates **pandas** for processing and analysis of uploaded dataset before passing context to the LLM.
- **Interactive Documentation:** Integrates **Swagger UI** for exploring, validating, and testing all API endpoints interactively.

## What the app does

1. Accepts a CSV file upload (e.g., The World Happiness Report 2019)
2. Reads and stores the dataset in memory through API.
3. Inspects and/or analyses the dataset through many API endpoints
4. Let a user ask natural language question about the dataset and get back an AI-generated answer based on the analysis statistics.

## Model

The app uses hosted Groq LLM
`model: llama-3.1-8b-instant`

## Dataset

The project was built and tested with the **World Happiness Report 2019** dataset from Kaggle.
Download: https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2019.csv

For the purpose of analysis, the dataset columns where renamed to a snake_case format

| Original name                | Renamed to      |
| ---------------------------- | --------------- |
| Overall rank                 | rank.           |
| Country or region            | country         |
| Score                        | happiness_score |
| GDP per capita               | gdp_per-capita  |
| Social support               | social_support  |
| Healthy life expectancy      | life_expectancy |
| Freedom to make life choices | freedom         |
| Generosity                   | generosity      |
| Perception of corruption     | corruption      |

## Project structure

```
├── .env
├── .gitignore
├── .python-version
├── .venv/
├── .vscode/
├── README.md
├── data-2019.csv
├── pyproject.toml
├── reflection.md
├── uv.lock
└── app/
    ├── __init__.py
    ├── config.py
    ├── data.py                 # In-memory dataset storage and conatins all analysis functions
    ├── main.py                 # Contains all endpoints, FastAPI app
    ├── schemas.py              # Pydantic models for API input/output and chain steps
    ├── chain/
    │   ├── __init__.py
    │   ├── pipeline.py         # Assembles the chain: oraklet : PromptBuilder() | LLMRunner() | ResponseParser()
    │   ├── runnable.py         # Contains Runnable base class, RunnableLambda, RunnableSequence
    │   └── steps.py            # Defines PromptBuilder, LLMRunner, ResponseParser
    └── tests/
        ├── __init__.py
        ├── conftest.py         # Contains shred fixtures: TestClient, sample CSV, dataset reset
        ├── test_chain.py       # Contains tests for each chain step in isolation
        └── test_endpoints.py   # Contains tests for all API endpoints
```

## Installations

This project uses `uv`for dependency management.

```bash
# Clone the gitHub repo and enter the folder
git clone git@github.com:Danse21/KK2-Oraklet-SmolLM.git
cd KK2-Oraklet-SmolLM

# Install dependencies
# Key dependencies: fastapi, uvicorn, numpy, pydantic, transformers
uv sync

# Get Groq API key
https://console.groq.com --> log in with Google or GitHub account --> click on API keys --> Create API Keys

Put the API key in a .env file (format: GROQ_API_KEY=put_your-key-here)

uv add groq python-dotenv
```

Because my computer device can only run on CPU which is slow for even small language models tested (SmolLM2-135M and 1.7B), this app is setup to use a hosted Groq API. As a result, `torch==2.2.2`, `transformers>=4.40,<5.0` was removed from `pyproject.toml`as there were no longer needed.

## Run the server

```bash
uv run uvicorn app.main:app --reload
```

The server starts at `http://localhost:800`.

Open `http://localhost:8000/docs` to go to interactive **Swagger UI** where you can explore the endpoints and analyze your data.

## Endpoints and example of expected response

`GET /health` - Health

```
Status code = 200
Response body:
{
"status": "ok"
}
```

`POST /data/upload`- Upload Data

```
Status code = 200
Response body:
{
  "rows": 156,
  "columns": [
    "rank",
    "country",
    "happiness_score",
    "gdp_per_capita",
    "social_support",
    "life_expectancy",
    "freedom",
    "generosity",
    "corruption"
  ],
  "dtypes": {
    "rank": "int64",
    "country": "str",
    "happiness_score": "float64",
    "gdp_per_capita": "float64",
    "social_support": "float64",
    "life_expectancy": "float64",
    "freedom": "float64",
    "generosity": "float64",
    "corruption": "float64"
  }
}
```

`GET /data/stats`- Data Stats

```
Status code = 200
Response body:
{
  "stats": {
    "happiness_score": {
      "count": 156,
      "mean": 5.407096153846155,
      "std": 1.1131198687956712,
      "min": 2.853,
      "25%": 4.5445,
      "50%": 5.3795,
      "75%": 6.1845,
      "max": 7.769
    },
    "gdp_per_capita": {
      "count": 156,
      "mean": 0.905147435897436,
      "std": 0.39838946424220284,
      "min": 0,
      "25%": 0.60275,
      "50%": 0.96,
      "75%": 1.2325000000000002,
      "max": 1.684
    },
    "social_support": {
      "count": 156,
      "mean": 1.2088141025641026,
      "std": 0.29919140069769296,
      "min": 0,
      "25%": 1.05575,
      "50%": 1.2715,
      "75%": 1.4525,
      "max": 1.624
    },
    "life_expectancy": {
      "count": 156,
      "mean": 0.7252435897435898,
      "std": 0.24212399840537246,
      "min": 0,
      "25%": 0.5477500000000001,
      "50%": 0.789,
      "75%": 0.88175,
      "max": 1.141
    },
    "freedom": {
      "count": 156,
      "mean": 0.39257051282051286,
      "std": 0.1432894707060473,
      "min": 0,
      "25%": 0.308,
      "50%": 0.417,
      "75%": 0.50725,
      "max": 0.631
    },
    "generosity": {
      "count": 156,
      "mean": 0.18484615384615383,
      "std": 0.09525444050922018,
      "min": 0,
      "25%": 0.10875,
      "50%": 0.1775,
      "75%": 0.24825,
      "max": 0.566
    },
    "corruption": {
      "count": 156,
      "mean": 0.11060256410256411,
      "std": 0.09453783536745279,
      "min": 0,
      "25%": 0.047,
      "50%": 0.08549999999999999,
      "75%": 0.14125,
      "max": 0.453
    }
  }
}
```

`GET /data/shape`- Data Shape

```
Status code = 200
Response body:
{
  "rows": 156,
  "columns": 9
}
```

`GET /data/top`- Data Top

```
Status code = 200
Response body:
{
  "results": [
    {
      "country": "Finland",
      "happiness_score": 7.769
    },
    {
      "country": "Denmark",
      "happiness_score": 7.6
    },
    {
      "country": "Norway",
      "happiness_score": 7.554
    },
    {
      "country": "Iceland",
      "happiness_score": 7.494
    },
    {
      "country": "Netherlands",
      "happiness_score": 7.488
    },
    {
      "country": "Switzerland",
      "happiness_score": 7.48
    },
    {
      "country": "Sweden",
      "happiness_score": 7.343
    },
    {
      "country": "New Zealand",
      "happiness_score": 7.307
    },
    {
      "country": "Canada",
      "happiness_score": 7.278
    },
    {
      "country": "Austria",
      "happiness_score": 7.246
    }
  ]
}
```

`GET /data/bottom`- Data Bottom

```
Status code = 200
Response body:
{
  "results": [
    {
      "country": "South Sudan",
      "happiness_score": 2.853
    },
    {
      "country": "Central African Republic",
      "happiness_score": 3.083
    },
    {
      "country": "Afghanistan",
      "happiness_score": 3.203
    },
    {
      "country": "Tanzania",
      "happiness_score": 3.231
    },
    {
      "country": "Rwanda",
      "happiness_score": 3.334
    }
  ]
}
```

`GET /data/zeros`- Data Zeros

```
Status code = 200
Response body:
{
"zeros": {
"gdp_per_capita": [
"Somalia"
],
"social_support": [
"Central African Republic"
],
"life_expectancy": [
"Swaziland"
],
"freedom": [
"Afghanistan"
],
"generosity": [
"Greece"
],
"corruption": [
"Moldova"
]
}
}
```

`GET /data/missing`- Data Missing

```
Status code = 200
Response body:
{
  "missing": {}
}
```

`GET /data/outliers`- Data Outliers

```
Status code = 200
Response body:
{
  "over_performers": [
    {
      "country": "Costa Rica",
      "happiness_score": 7.167
    },
    {
      "country": "Finland",
      "happiness_score": 7.769
    },
    {
      "country": "Somalia",
      "happiness_score": 4.668
    },
    {
      "country": "Guatemala",
      "happiness_score": 6.436
    },
    {
      "country": "Nicaragua",
      "happiness_score": 6.105
    }
  ],
  "under_performers": [
    {
      "country": "Botswana",
      "happiness_score": 3.488
    },
    {
      "country": "Syria",
      "happiness_score": 3.462
    },
    {
      "country": "Iran",
      "happiness_score": 4.548
    },
    {
      "country": "Iraq",
      "happiness_score": 4.437
    },
    {
      "country": "Egypt",
      "happiness_score": 4.166
    }
  ]
}
```

`GET /data/correlations`- Data Correlations

```
Status code = 200
Response body:
{
  "correlations": {
    "gdp_per_capita": 0.794,
    "life_expectancy": 0.78,
    "social_support": 0.777,
    "freedom": 0.567,
    "corruption": 0.386,
    "generosity": 0.076
  }
}
```

`GET /ai/ask`- Ask

```
Request body:
{
"question": "Which factor contributes most to happiness score?"
}
```

**Note** that the "question" field must be between 5 and 500 characters to be accepted.

**Example server response**

```

Status code = 200
Response body:
{
"question": "Which factor contributes most to happiness score?",
"answer": "Based on the factor correlations with happiness score, the factor that contributes most to happiness score is gdp_per_capita with a correlation of 0.794.",
"model": "llama-3.1-8b-instant"
}

```

## Running the tests

```bash
# Add pytest to your project
uv add pytest --dev

# Run all tests at once
uv run pytest app/tests/ -v

# Run only chain step tests
uv run pytest app/tests/test_chain.py -v

# Run only endpoint tests
uv run pytest app/tests/test_endpoints.py -v
```
