from fastapi import FastAPI
from app.ml.main import predict
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

model_name = "Food Image Recognition"
version = "v1.0.0"
key = os.getenv("API_KEY")
csv_path = "app/data/kandungan_buah_sayur.csv"
df = pd.read_csv(csv_path)

@app.get("/")
def read_root():
    return {"Welcome"}

@app.get('/info')
async def model_info():
    """ Return model information and version """
    return {
        "name": model_name,
        "version": version
    }

@app.post('/recognize')
async def recognize_image(image: UploadFile, api_key: str):
    """ Recognize the uploaded image """
    if not api_key or not image:
        raise HTTPException(status_code=400, detail="Please provide an API key and an image")
    if "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="File must be an image")
    if api_key != key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    img = Image.open(image.file)
    predicted_class, confidence = predict(img)
    filtered_df = df[df['nama'].str.lower() == predicted_class]
    if filtered_df.empty:
        result_data = {}
    else:
        result_data = filtered_df.iloc[0].to_dict()
    return {
        "name": model_name,
        "version": version,
        "result": result_data,
        "confidence": str(confidence)
    }