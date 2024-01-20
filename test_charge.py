from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between requests

    @task
    def predict_endpoint(self):
        # Define your request to the prediction endpoint
        self.client.post("https://group3-container.salmonwater-c5230928.francecentral.azurecontainerapps.io/predict", json={"features": [5.1, 3.5,1.4, 0.2]})

# Create 10 users
class WebsiteUser(HttpUser):
    tasks = [MyUser] * 10  # Simulate 10 users
    wait_time = between(1, 5)  # Wait time between users
