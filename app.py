from pathlib import Path
from typing import Annotated, Literal
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "model.pkl"

# Import the ml model
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

app = FastAPI()

tier_1_cities = [
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad"
]

tier_2_cities = [
    "Jaipur",
    "Lucknow"
]

# pydantic model to validate incoming data
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the user")]
    weight: Annotated[float, Field(..., gt=0, description="Weight of the user")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height of the user")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual Salary of the user in lpa")]
    smoker: Annotated[bool, Field(..., description="Whether the user smokes")]
    city: Annotated[str, Field(..., description="City of the user")]
    occupation: Annotated[Literal['Doctor', 'Teacher', 'Sales Executive', 'Data Analyst', 'Nurse',
       'Government Employee', 'Software Engineer', 'Student',
       'Mechanical Engineer', 'Business Owner', 'Accountant', 'Lawyer'], Field(..., description="Occupation of the user")]


    # create computed fields for calculations
    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight/(self.height**2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker and self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age<25:
            return "young"
        elif self.age<45:
            return "adult"
        elif self.age<60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else: 
            return 3


@app.post('/predict')
def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'predicted_category': prediction})
