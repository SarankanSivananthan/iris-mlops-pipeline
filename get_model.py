import numpy as np
from collections import OrderedDict
import joblib
import time
from flask import Flask, request, Response, jsonify
from prometheus_client import Counter, Gauge, generate_latest, REGISTRY
 
app = Flask(__name__)

# Counter pour le nombre total de requêtes
api_call_counter = Counter('api_calls_total', 'Total number of API calls')

# Load the trained model
model = joblib.load('./model.pkl')

def species_mapping(prediction):
    if prediction == float(0) :
        prediction = "Iris setosa"
    elif prediction == float(1) :
        prediction = "Iris versicolor" 
    else :
        prediction = "Iris virginica"
    return prediction

@app.route('/predict', methods=["POST"])
def predict():
    global last_request_timestamp

    try:
        # Get input data from the request
        sepal_length = request.args.get('sepal_length')
        sepal_width = request.args.get('sepal_width')
        petal_length = request.args.get('petal_length')
        petal_width = request.args.get('petal_width')

        # Assuming the input data is in the same format as the Iris dataset
        features = np.array([float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)
        prediction = species_mapping(prediction)

        api_call_counter.inc()

        response_dict = OrderedDict([
            ('Your Input Features', OrderedDict([
                ('Sepal Length', sepal_length),
                ('Sepal Width', sepal_width),
                ('Petal Length', petal_length),
                ('Petal Width', petal_width),
            ])),
            ('Predicted Specie', prediction)
        ])

        return jsonify(response_dict)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/metrics')
def metrics():
    # Expose the metrics in Prometheus format
    return Response(generate_latest(REGISTRY), content_type='text/plain; version=0.0.4')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
