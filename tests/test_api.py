import re

import pytest

from get_model import app


@pytest.fixture
def client():
    return app.test_client()


def predict(client, sepal_length, sepal_width, petal_length, petal_width):
    return client.post(
        "/predict",
        query_string={
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    )


def api_calls_total(client):
    body = client.get("/metrics").get_data(as_text=True)
    match = re.search(r"^api_calls_total (\S+)$", body, re.MULTILINE)
    assert match, "api_calls_total missing from /metrics"
    return float(match.group(1))


@pytest.mark.parametrize(
    "features, species",
    [
        ((5.1, 3.5, 1.4, 0.2), "Iris setosa"),
        ((5.9, 2.8, 4.3, 1.3), "Iris versicolor"),
        ((6.7, 3.0, 5.2, 2.3), "Iris virginica"),
    ],
)
def test_predict_returns_the_right_species(client, features, species):
    response = predict(client, *features)
    assert response.status_code == 200
    assert response.get_json()["Predicted Specie"] == species


def test_predict_echoes_the_input_features(client):
    echoed = predict(client, 5.1, 3.5, 1.4, 0.2).get_json()["Your Input Features"]
    assert echoed == {
        "Sepal Length": "5.1",
        "Sepal Width": "3.5",
        "Petal Length": "1.4",
        "Petal Width": "0.2",
    }


def test_predict_without_features_returns_an_error_message(client):
    response = client.post("/predict")
    assert "error" in response.get_json()


def test_metrics_endpoint_exposes_the_call_counter(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_calls_total" in response.get_data(as_text=True)


def test_each_successful_prediction_increments_the_counter(client):
    before = api_calls_total(client)
    predict(client, 5.1, 3.5, 1.4, 0.2)
    predict(client, 6.7, 3.0, 5.2, 2.3)
    assert api_calls_total(client) == before + 2
