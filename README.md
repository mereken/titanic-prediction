# Titanic Survival Prediction: FastAPI + Docker

A machine learning model (Random Forest) trained on the Titanic dataset
and served as a REST API with FastAPI. The app is containerized with Docker.

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app
├── train.py           # trains and saves the model
├── model.joblib       # saved model (sklearn Pipeline)
├── requirements.txt   # dependencies
├── Dockerfile         # container instructions
└── README.md
```

## Model

- Dataset: Titanic (loaded via seaborn)
- Features: pclass, sex, age, sibsp, parch, fare, embarked
- Model: Random Forest inside an sklearn Pipeline (imputation + one-hot encoding + classifier)
- Target: survived (0 / 1)

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
python train.py                # creates model.joblib
uvicorn main:app --reload
```

Open:
- http://127.0.0.1:8000/       - health check
- http://127.0.0.1:8000/docs   - Swagger UI

## Run with Docker

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

## API

### GET /
```json
{"message": "ML API is running"}
```

### POST /predict
Request:
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```
Response (example):
```json
{
  "prediction": 0,
  "survived": false,
  "survival_probability": 0.12
}
```

