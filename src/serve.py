from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# AWS S3 Configuration
S3_BUCKET = "mlops-hieu-202608211828"
MODEL_PATH = "/tmp/model.pkl"


def download_model():
    """Tai model tu S3."""
    print(f"Dang tai model tu S3 bucket: {S3_BUCKET}")
    s3 = boto3.client("s3")
    os.makedirs("/tmp", exist_ok=True)
    s3.download_file(S3_BUCKET, "models/latest/model.pkl", MODEL_PATH)
    print(f"Da tai model: {MODEL_PATH}")


# Tai model khi server khoi dong
download_model()
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")


class PredictRequest(BaseModel):
    features: list[float]


LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.

    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    Dau ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)"
        )

    prediction = model.predict([req.features])[0]
    label = LABELS.get(int(prediction), "unknown")

    return {
        "prediction": int(prediction),
        "label": label
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
