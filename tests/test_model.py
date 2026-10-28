import joblib
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

import model as model_script

# Canonical rows from the Iris dataset, one per species.
SETOSA = [5.1, 3.5, 1.4, 0.2]
VERSICOLOR = [5.9, 2.8, 4.3, 1.3]
VIRGINICA = [6.7, 3.0, 5.2, 2.3]


def test_train_returns_scaler_and_classifier_in_one_pipeline():
    pipeline, _, _ = model_script.train()
    assert [name for name, _ in pipeline.steps] == ["scaler", "knn"]


def test_train_reaches_reasonable_accuracy():
    _, metrics, _ = model_script.train()
    assert metrics["accuracy"] >= 0.9
    assert metrics["f1_macro"] >= 0.9


def test_n_neighbors_is_applied():
    pipeline, _, _ = model_script.train(n_neighbors=7)
    assert pipeline.named_steps["knn"].n_neighbors == 7


def test_pipeline_classifies_raw_measurements():
    # The API sends raw centimetres, so the exported model has to standardise
    # them itself. A bare KNN trained on scaled data predicts one class for
    # everything here.
    pipeline, _, _ = model_script.train()
    predictions = pipeline.predict(np.array([SETOSA, VERSICOLOR, VIRGINICA]))
    assert predictions.tolist() == [0, 1, 2]


def test_train_and_track_logs_params_metrics_and_model(tmp_path, monkeypatch):
    # MLflow's SQLite store creates a default ./mlruns in the working directory,
    # so run from the temp dir to keep the repository clean.
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "test-iris-knn", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    _, metrics, run_id = model_script.train_and_track(
        n_neighbors=3, model_path=str(tmp_path / "model.pkl")
    )

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["n_neighbors"] == "3"
    assert run.data.params["model_type"] == "KNeighborsClassifier"
    assert run.data.metrics["accuracy"] == pytest.approx(metrics["accuracy"])
    assert run.data.metrics["f1_macro"] == pytest.approx(metrics["f1_macro"])

    logged = [artifact.path for artifact in client.list_artifacts(run_id, "model")]
    assert "model/MLmodel" in logged
    assert (tmp_path / "model.pkl").exists()


def test_committed_model_pkl_is_the_scaled_pipeline():
    # model.pkl is what the Docker image ships and the API serves.
    served = joblib.load("model.pkl")
    assert [name for name, _ in served.steps] == ["scaler", "knn"]
    assert served.predict(np.array([SETOSA, VERSICOLOR, VIRGINICA])).tolist() == [0, 1, 2]
