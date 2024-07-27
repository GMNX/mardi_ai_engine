import os
import sys
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pika import BasicProperties

import torch

from bridge import get_rabbitmq_connection
from crud import create_inference, get_inference, delete_inference
from database import SessionLocal, init_db
from schemas import InferenceCreate, Inference

# Initialize API Server
app = FastAPI(
    title="Mardi AI Inference API",
    description="API for performing inference on images",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
def on_startup():
    init_db()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post('/detect', response_model=Inference)
def do_detect(payload: InferenceCreate, db: Session = Depends(get_db)):
    """
    Perform prediction on input data
    """
    new_inference = create_inference(db, payload)
    rabbitmq_client = get_rabbitmq_connection()
    rabbitmq_client.basic_publish(
        exchange='',  # default exchange
        routing_key='mardi_inference_queue',
        body=new_inference.image_url,
        properties=BasicProperties(headers={'inference_id': new_inference.id})
    )

    return new_inference


@app.get('/status/{inference_id}', response_model=Inference)
def fetch_inference(inference_id: str, db: Session = Depends(get_db)):
    """
    Get inference by ID
    """
    result = get_inference(db, inference_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference not found")

    return result


@app.delete('/{inference_id}')
def remove_inference(inference_id: str, db: Session = Depends(get_db)):
    """
    Delete inference by ID
    """
    result = delete_inference(db, inference_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference not found")

    return {"id": result.id,"status": "Deleted"}


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True)
