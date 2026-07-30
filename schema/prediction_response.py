from pydantic import BaseModel, Field
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",
        examples=["High"],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence score for the predicted class",
        examples=[0.82],
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability score for each premium category",
        examples=[{"High": 0.82, "Low": 0.05, "Medium": 0.13}],
    )
