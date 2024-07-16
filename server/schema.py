'''Inference schema for the server'''
from typing import Optional
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    image_url: str = Field(..., title="Image URL", description="URL of the image to be classified")


class InferenceResponse(BaseModel):
    """
    Output values for model inference
    """
    status: str = Field(..., title="Status", description="Status of the inference")
    age: Optional[str] = Field(None, title="Age",
                               description="Predicted age of the plant in the image")
    image_result: Optional[str] = Field(
        None,title="Image Result",
        description="Result of the image classification in base64 format")


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')