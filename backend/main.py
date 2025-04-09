from fastapi import FastAPI
from backend.routes import predict_skin

app = FastAPI()

app.include_router(predict_skin.router)

@app.get("/")
def read_root():
    return {"message": "Smart Health Prediction API is running!"}
