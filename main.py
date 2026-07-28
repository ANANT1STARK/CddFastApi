import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import json
import io

load_dotenv()  # reads variables from a local .env file

app = FastAPI()

# Allow requests from the React Native app during development.
# Tighten this before deploying (restrict to your actual app's origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model + class mapping ----
MODEL_PATH = os.getenv("MODEL_PATH")
CLASS_MAPPING_PATH = os.getenv("CLASS_MAPPING_PATH")

model = load_model(MODEL_PATH)

with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = {int(k): v for k, v in json.load(f).items()}

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

# ---- MongoDB connection ----
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "crop_disease_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "diseases")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
diseases_collection = db[COLLECTION_NAME]

# Maps the label your model/class_mapping.json outputs to the disease_id used in MongoDB.
# IMPORTANT: check the actual string values inside class_mapping_6class.json and update
# the keys below to match exactly (case-sensitive) — this is a best guess based on your
# 5 trained diseases + a "Not Cashew" class.
CLASS_TO_DISEASE_ID = {
    "Cashew gumosis": "gumosis",
    "Cashew anthracnose": "anthracnose",
    "Cashew leaf miner": "leaf_miner",
    "Cashew red rust": "red_dust",
    "Cashew healthy": "healthy",
}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")

    image_bytes = await file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image - file may be corrupted or unsupported")

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_mapping[predicted_index]
    confidence = float(predictions[0][predicted_index])

    if predicted_class == "Not Cashew":
        return {
            "status": "not_cashew",
            "predicted_class": "Not Cashew",
            "confidence": round(confidence, 4),
            "message": "This doesn't appear to be a cashew leaf.",
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "low_confidence",
            "predicted_class": "uncertain",
            "confidence": round(confidence, 4),
            "message": "Low confidence - image may be unclear or not a cashew leaf",
            "top_guess": predicted_class,
        }

    # Confident prediction — look up the full disease info from MongoDB.
    disease_id = CLASS_TO_DISEASE_ID.get(predicted_class)

    if not disease_id:
        raise HTTPException(
            status_code=500,
            detail=f"No disease_id mapping found for predicted class '{predicted_class}'",
        )

    db_result = await diseases_collection.find_one({"disease_id": disease_id})

    if not db_result:
        raise HTTPException(
            status_code=404,
            detail=f"No database entry found for disease_id '{disease_id}'",
        )

    db_result["_id"] = str(db_result["_id"])
    db_result["confidence"] = round(confidence, 4)
    db_result["status"] = "confident"

    return db_result


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Cashew disease detection API is running"}