'''Inference schema for the server'''
from datetime import datetime
from collections import OrderedDict
from typing import Optional
from pydantic import BaseModel, Field

class InferenceCreate(BaseModel):
    """
    Input values for model inference
    """
    image_url: str = Field(..., title="Image URL", description="URL of the image to be classified")


class Inference(InferenceCreate):
    """
    Output values for model inference
    """
    id: str = Field(..., title="Inference ID", description="Unique identifier for the inference")
    status: str = Field(..., title="Status", description="Status of the inference")
    age: Optional[str] = Field(None, title="Age",
                               description="Predicted age of the plant in the image")
    image_result: Optional[str] = Field(
        None,title="Image Result",
        description="Result of the image classification in base64 format")
    progress: Optional[float] = Field(None, title="Progress", description="Progress of the inference")
    created_at: datetime = Field(..., title="Created At", description="Date and time the inference was created")
    updated_at: datetime = Field(..., title="Updated At", description="Date and time the inference was last updated")

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        return OrderedDict((k, d[k]) for k in self.__annotations__.keys())

    class Config:
        orm_mode: True

    def __str__(self):
        return f"Inference(id={self.id}, status={self.status}, age={self.age}, image_result={self.image_result}, progress={self.progress}, created_at={self.created_at}, updated_at={self.updated_at})"

    def __repr__(self):
        return self.__str__()


class InferenceUpdate(BaseModel):
    """
    Input values for updating an inference record
    """
    status: Optional[str] = Field(None, title="Status", description="Status of the inference")
    age: Optional[str] = Field(None, title="Age", description="Predicted age of the plant in the image")
    image_result: Optional[str] = Field(None, title="Image Result",
                                        description="Result of the image classification in base64 format")
    progress: Optional[float] = Field(None, title="Progress", description="Progress of the inference")
