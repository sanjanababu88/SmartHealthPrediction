from fastapi import APIRouter, File, UploadFile
import shutil
import os
from backend.models.skin_disease.model import predict_skin_disease

router = APIRouter()

UPLOAD_DIR = "backend/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_skin_disease(file_path)
    
    return {"prediction": result}
