"""Train the Iris classifier, track the run with MLflow, and export model.pkl for the API.

    python model.py                    # train with the defaults (k=5)
    python model.py --n-neighbors 7    # a second run to compare in the MLflow UI

Every run logs its parameters, metrics and the fitted model to MLflow. Browse
and compare runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_NAME = "iris-knn"
MODEL_PATH = "./model.pkl"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def train(n_neighbors=5, test_size=0.2, random_state=25):
    """Fit scaler + KNN on a train split and score them on the held-out split.

    The scaler and the classifier are kept in a single Pipeline on purpose:
    the model is trained on standardised features, so whatever serves it must
    apply the exact same standardisation. Exporting the classifier alone would
    make the API feed raw centimetres to a model that expects z-scores.
    """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    return pipeline, metrics, X_train


def train_and_track(n_neighbors=5, test_size=0.2, random_state=25, model_path=MODEL_PATH):
    """Train inside an MLflow run (params, metrics, model) and write `model_path` for the API."""
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": "KNeighborsClassifier",
            "scaler": "StandardScaler",
            "n_neighbors": n_neighbors,
            "test_size": test_size,
            "random_state": random_state,
        })

        pipeline, metrics, X_train = train(n_neighbors, test_size, random_state)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train[:3],
        )

        joblib.dump(pipeline, model_path)
        return pipeline, metrics, run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=25)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    _, metrics, run_id = train_and_track(args.n_neighbors, args.test_size, args.random_state)
    print(f"MLflow run: {run_id}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Model written to {MODEL_PATH}")


if __name__ == "__main__":
    main()
