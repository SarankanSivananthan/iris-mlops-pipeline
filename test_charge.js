import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  vus: 100,
  duration: '50s',
};

export default function () {
  const url = 'https://group3-container.bluesmoke-5ac04595.francecentral.azurecontainerapps.io/predict';

  const payload = {
    sepal_length: 5.1,
    sepal_width: 3.5,
    petal_length: 1.4,
    petal_width: 0.2
  };

  const headers = {
    'Content-Type': 'application/json'
  };

  const response = http.post(url, JSON.stringify(payload), { headers: headers });

  sleep(1);
}