import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-survival")
MODEL_NAME = "titanic-survival-model"

df = sns.load_dataset("titanic")

NUM_FEATURES = ["age", "sibsp", "parch", "fare"]
CAT_FEATURES = ["pclass", "sex", "embarked"]

X = df[NUM_FEATURES + CAT_FEATURES]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def build_model(n_estimators, max_depth):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )),
    ])


param_grid = [
    {"n_estimators": n, "max_depth": d}
    for n in (50, 100, 200)
    for d in (3, 5, 8)
]

best = {"f1": -1.0, "model": None, "uri": None, "run_id": None}

for params in param_grid:
    run_name = f"rf_n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
        }

        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("dataset", "titanic")

        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.joblib")
            joblib.dump(model, path)
            mlflow.log_artifact(path)

        print(f"{run_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

        if metrics["f1_score"] > best["f1"]:
            best.update(f1=metrics["f1_score"], model=model,
                        uri=model_info.model_uri, run_id=run.info.run_id)

joblib.dump(best["model"], "model.joblib")
print(f"\nBest run: {best['run_id']} (f1={best['f1']:.4f}) -> saved to model.joblib")

result = mlflow.register_model(best["uri"], MODEL_NAME)
client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
print(f"Registered model: name={result.name}, version={result.version}, alias=champion")
