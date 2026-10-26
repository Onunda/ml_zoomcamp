import os
import sys

import requests


# 1. Tee class configured with append mode ("a") to preserve existing logs
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


os.makedirs("./output", exist_ok=True)
sys.stdout = Tee("./output/homework5_output.txt")

# 3. Define prediction test request
url = "http://localhost:9696/predict"

lead = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
}

print("Sending request to FastAPI prediction endpoint...")
response = requests.post(url, json=lead)
print(f"Response JSON: {response.json()}")
