import os
import sys
import random
import traceback

import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

import torch

from utils import add_text_to_image
from schema import InferenceInput, InferenceResponse, ErrorResponse

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


@app.post('/detect',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_detect(body: InferenceInput):
    """
    Perform prediction on input data
    """
    try:
        # Get the input data
        image_url = body.image_url
        text = "Detected"

        # Perform the prediction
        result = add_text_to_image(image_url, text)
        age_choice = ['week1', 'week2', 'week3', 'week4']
        age = random.choice(age_choice)

        return InferenceResponse(status="success", age=age, image_result=result)

    except Exception as e:
        logger.error(traceback.format_exc())
        return ErrorResponse(error=True, message=str(e), traceback=traceback.format_exc())


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
