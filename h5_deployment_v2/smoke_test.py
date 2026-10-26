import os
import pickle
import sys


class Termi:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs("output", exist_ok=True)
sys.stdout = Termi("output/homework5_output.txt")


def run_smoke_test():
    print("Loading pipeline.bin...")
    with open("pipeline.bin", "rb") as f_in:
        pipeline = pickle.load(f_in)

    test_lead = {
        "lead_source": "paid_ads",
        "industry": "technology",
        "employment_status": "employed",
        "location": "north_america",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
        "interaction_count": 4,
        "lead_score": 0.41
    }

    print("Running test prediction...")
    prediction = int(pipeline.predict([test_lead])[0])
    probabilities = pipeline.predict_proba([test_lead])[0]
    conversion_probability = float(probabilities[1])

    print("Smoke test passed successfully!")
    print(f"-> Prediction: {prediction}")
    print(f"-> Conversion Probability: {conversion_probability:.3f}")


if __name__ == "__main__":
    run_smoke_test()
