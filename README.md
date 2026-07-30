# Insurance Premium Prediction FastAPI

A small machine learning project that predicts an insurance premium category from user details. The API is built with FastAPI, and the model is trained with scikit-learn.

## Project Structure

```text
.
|-- app.py
|-- insurance_premium.py
|-- data/
|   `-- insurance_premium_dataset_200_rows.csv
|-- trained_model/
|   `-- model.pkl
|-- requirements.txt
|-- pyproject.toml
`-- uv.lock
```

## Requirements

- Python 3.10
- uv

## Setup

Create a Python 3.10 virtual environment:

```powershell
uv venv --python 3.10
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

## Run the FastAPI Backend

```powershell
uvicorn app:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

The prediction endpoint is:

```text
POST /predict
```

Example request body:

```json
{
  "age": 30,
  "weight": 65,
  "height": 1.7,
  "income_lpa": 10,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "Software Engineer"
}
```

## Retrain the Model

```powershell
python insurance_premium.py
```

This reads the dataset from `data/insurance_premium_dataset_200_rows.csv` and saves the trained model to `trained_model/model.pkl`.

## Notes

- User height in the API and Streamlit app is entered in meters.
- Dataset height is stored in centimeters and converted to meters during training.
- Current model accuracy on the existing train/test split is around 62.50%.
