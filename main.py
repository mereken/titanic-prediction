from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival API")

model = joblib.load("model.joblib")


class PassengerInput(BaseModel):
    pclass: Literal[1, 2, 3] = Field(..., description="Passenger class (1, 2 or 3)")
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=100)
    sibsp: int = Field(..., ge=0, description="Siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Parents/children aboard")
    fare: float = Field(..., ge=0, description="Ticket fare")
    embarked: Literal["S", "C", "Q"] = Field(..., description="Port of embarkation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S",
            }
        }
    }


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: PassengerInput):
    df = pd.DataFrame([data.model_dump()])
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    return {
        "prediction": pred,
        "survived": bool(pred),
        "survival_probability": round(proba, 4),
    }
