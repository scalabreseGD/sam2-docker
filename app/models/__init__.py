from typing import Literal, Optional, List, Union

from pydantic import BaseModel, Field

SAM2_MODELS = ("sam2.1_hiera_tiny", "sam2.1-hiera-small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large")


class PredictArgs(BaseModel):
    model: Literal[SAM2_MODELS] = Field(...,
                                        description="The models to use between these values\n" + '\n'.join(
                                            SAM2_MODELS))
    images: Optional[List[str]] = Field(...,
                                        description="The images to predict in base64 or the path of the images to load")
    video: Optional[str] = Field(..., description="The path of the video to predict")
    scale_factor: Optional[float] = Field(1, description="The scale factor of the media to reduce the memory")
    start_second: Optional[int] = Field(0, description="The starting frame for the prediction")
    end_second: Optional[int] = Field(None, description="The end frame for the prediction")


class PredictResponse(BaseModel):
    response: Optional[Union[str, dict]] = Field(..., description="The output from Sam2")
