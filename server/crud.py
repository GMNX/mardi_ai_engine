from sqlalchemy.orm import Session
from models import Inference
from schemas import InferenceCreate, InferenceUpdate

def create_inference(db: Session, inference: InferenceCreate):
    """
    Create a new inference record
    """
    db_inference = Inference(
        image_url=inference.image_url,
        status="pending"
    )
    db.add(db_inference)
    db.commit()
    db.refresh(db_inference)
    return db_inference

def get_inference(db: Session, inference_id: str):
    """
    Get an inference record by ID
    """
    return db.query(Inference).filter(Inference.id == inference_id).first()

def update_inference(db: Session, inference_id: str, inference: InferenceUpdate):
    """
    Update an inference record by ID
    """
    db_inference = db.query(Inference).filter(Inference.id == inference_id).first()
    for key, value in inference.model_dump().items():
        setattr(db_inference, key, value)
    db.commit()
    db.refresh(db_inference)
    return db_inference

def delete_inference(db: Session, inference_id: str):
    """
    Delete an inference record by ID
    """
    db_inference = db.query(Inference).filter(Inference.id == inference_id).first()
    db.delete(db_inference)
    db.commit()
    return db_inference
