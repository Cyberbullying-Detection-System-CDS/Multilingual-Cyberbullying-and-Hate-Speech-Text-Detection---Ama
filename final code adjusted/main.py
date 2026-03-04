# main.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# -------------------------------
# Logging (helps debugging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hate Speech / Cyberbullying Detection API")

MODEL_PATH = r"F:\SLIIT bachelors degree\Research Project (RP)\hatespeech-detection\cyberbullying_xlmr_model"

# Load model & tokenizer with error handling
try:
    logger.info(f"Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logger.info(f"Loading model from {MODEL_PATH} ...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")

except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}", exc_info=True)
    raise RuntimeError("Model loading failed. Check MODEL_PATH and files.")


class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float          # 0–100%
    is_cyberbullying: bool

# -------------------------------
def predict(text: str) -> dict:
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item() * 100  # percentage

        label = "Cyberbullying" if pred_class == 1 else "Non-Cyberbullying"
        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "is_cyberbullying": pred_class == 1
        }

    except Exception as e:
        logger.error(f"Prediction failed for text: '{text}' → {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = predict(request.text)
    return result

@app.get("/")
async def root():
    return {
        "message": "Hate Speech Detection API is running",
        "model_path": MODEL_PATH,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)