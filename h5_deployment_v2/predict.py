import json
import pickle

from fastapi import FastAPI, Request

app = FastAPI(title="Lead Scoring Prediction Service")

with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

metadata = {"data_checksum": "unknown"}
try:
    with open("model_metadata.json", "r") as f_meta:
        content = f_meta.read().strip()
        if content:
            metadata = json.loads(content)
except (FileNotFoundError, json.JSONDecodeError):
    pass


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "data_checksum": metadata.get("data_checksum")
    }


@app.post("/predict")
async def predict(request: Request):
    lead_data = await request.json()
    X = [lead_data]
    probabilities = pipeline.predict_proba(X)[0]
    conversion_probability = float(probabilities[1])
    prediction = int(pipeline.predict(X)[0])
    return {
        "prediction": prediction,
        "conversion_probability": round(conversion_probability, 6)
    }
