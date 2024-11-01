from typing import Literal, Optional, List, Union, Tuple

from pydantic import BaseModel, Field

SAM2_MODELS = ("sam2.1-hiera-tiny", "sam2.1-hiera-small", "sam2.1-hiera-base-plus", "sam2.1-hiera-large")


class BoxOrPoint(BaseModel):
    frame: int = Field(..., description="frame index")
    object_id: int = Field(..., description="object id to be annotated")
    bbox: Optional[Tuple[float, float, float, float]] = Field((0.0, 0.0, 0.0, 0.0),
                                                              description="The bounding box in x1,y1,x2,y2 format")
    point: Optional[Tuple[float, float]] = Field((0.0, 0.0),
                                                 description="The anchor point in x,y format")
    label: Literal[0, 1] = Field(..., description="label indicating if the point is to be excluded or included in 0/1 format")


class PredictArgs(BaseModel):
    model: Literal[SAM2_MODELS] = Field(...,
                                        description="The models to use between these values\n" + '\n'.join(
                                            SAM2_MODELS))
    images: Optional[List[str]] = Field(...,
                                        description="The images to predict in base64 or the path of the images to load")
    video: Optional[str] = Field(..., description="The path of the video to predict")
    boxOrPoint: List[BoxOrPoint] = Field(...,
                                         description="Boxes or points to be assigned in the video or photo. Must be at least one")
    scale_factor: Optional[float] = Field(1, description="The scale factor of the media to reduce the memory")
    start_second: Optional[int] = Field(0, description="The starting frame for the prediction")
    end_second: Optional[int] = Field(None, description="The end frame for the prediction")


class PredictResponse(BaseModel):
    response: dict[int, List] = Field(..., description="The output masks from Sam2")
