import uuid
from datetime import datetime
import pytz
from sqlalchemy import Column, String, Float, DateTime, event
from sqlalchemy.sql import func
from database import Base

class Inference(Base):
    __tablename__ = "inferences"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    image_url = Column(String, nullable=False)
    image_result = Column(String, nullable=True)
    age = Column(String, nullable=True)
    status = Column(String, nullable=True)
    progress = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    def __str__(self):
        return f"Inference(id={self.id}, status={self.status}, age={self.age}, image_result={self.image_result}, progress={self.progress}, created_at={self.created_at}, updated_at={self.updated_at})"
    
    def __repr__(self):
        return self.__str__()

gmt_plus_8 = pytz.timezone('Asia/Kuala_Lumpur')

# Event listener to set created_at and updated_at before insert
@event.listens_for(Inference, 'before_insert')
def set_created_at(mapper, connection, target):
    now = datetime.now(gmt_plus_8)
    target.created_at = now
    target.updated_at = now

# Event listener to set updated_at before update
@event.listens_for(Inference, 'before_update')
def set_updated_at(mapper, connection, target):
    now = datetime.now(gmt_plus_8)
    target.updated_at = now
