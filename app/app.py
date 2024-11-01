import os
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response

from api import florence, file_uploader
from middleware import LimitRequestSizeMiddleware, lifespan
from models import PredictArgs, PredictResponse

app = FastAPI(lifespan=lifespan)
app.add_middleware(LimitRequestSizeMiddleware)


def __return_response(request: PredictArgs) -> List[PredictResponse]:
    if request.video is not None and request.images is not None:
        Response(
            "Cannot use both images and video in the same request", status_code=400
        )
    model = florence(request.model)
    responses = model.call_model(task=request.task,
                                 text=request.text,
                                 images=request.images,
                                 video=request.video,
                                 batch_size=request.batch_size,
                                 scale_factor=request.scale_factor,
                                 start_second=request.start_second,
                                 end_second=request.end_second)
    return [PredictResponse(task=request.task, response=resp[request.task]) for resp in responses]


@app.post("/v1/predict", response_model=List[PredictResponse])
async def predict(request: PredictArgs):
    return __return_response(request)


@app.put("/v1/asset")
async def asset(files: List[UploadFile] = File(...)):
    return file_uploader().upload_batch(files)


if __name__ == "__main__":
    port = os.environ.get('PORT', '8000')
    uvicorn.run(app, host="0.0.0.0", port=int(port), log_level="info")
