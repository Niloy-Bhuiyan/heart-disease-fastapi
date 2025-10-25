from fastapi import FastAPI
from app.schemas import HeartInput
from app.model_loader import load_model

app = FastAPI(title="Heart Disease Prediction API")

model = load_model()

# Root route
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_type": str(type(model).__name__),
        "features": ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]
    }

@app.post("/predict")
def predict(data: HeartInput):
    values = [data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
              data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
              data.ca, data.thal]
    prediction = model.predict([values])[0]
    return {"heart_disease": bool(int(prediction))}
