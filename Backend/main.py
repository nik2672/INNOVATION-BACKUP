# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neural import predict_price, load_or_train_model

app = FastAPI()

# CORS configuration to allow requests from the frontend
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model weights or train if weights are missing
load_or_train_model()

# Request model for the API endpoint
class PredictionRequest(BaseModel):
    Rooms: int
    Type: str  # "h" for house, "t" for townhouse, "u" for unit
    Postcode: int
    Distance: float

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionRequest):
    # Call predict_price with the user's inputs
    predicted_price = predict_price(data.Rooms, data.Type, data.Postcode, data.Distance)
    # Convert numpy float to Python float to avoid JSON serialization errors
    return {"predicted_price": float(predicted_price)}
