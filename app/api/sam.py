import os
from typing import Union, List

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from api.utils import load_video_from_path, is_base64_string, base64_to_image_with_size, load_image_from_path
from api.patches import DEVICE
from tqdm import tqdm

from models import BoxOrPoint, MaskResponse


class SAM2:
    def __init__(self, model_name,
                 hf_token=None):
        self.hf_token = hf_token
        self.model_name = f"facebook/{model_name}"
        self.device = DEVICE
        self.sam2_model: Union[SAM2ImagePredictor, SAM2VideoPredictor, None] = None
        self.inference_state = None

    def __init_model(self, images=None, images_path=None):
        if images is None and images_path is None:
            raise ValueError("Either images or images_path must be set")

        if not self.sam2_model and images_path:
            self.sam2_model = SAM2VideoPredictor.from_pretrained(self.model_name, device=self.device.type,
                                                                 token=self.hf_token)
            if self.inference_state is not None:
                self.sam2_model.reset_state(self.inference_state)

            self.inference_state = self.sam2_model.init_state(video_path=images_path, offload_video_to_cpu=True,
                                                              offload_state_to_cpu=True
                                                              )
        else:
            self.sam2_model = SAM2ImagePredictor.from_pretrained(self.model_name, device=self.device.type,
                                                                 token=self.hf_token)

    def call_model(self, images, video, boxOrPoint, scale_factor,
                   start_second, end_second):
        if video is not None:
            images_pillow_with_size = load_video_from_path(video, scale_factor, start_second, end_second)
        elif images is not None and is_base64_string(images[0]):
            images_pillow = [base64_to_image_with_size(image)[0] for image in images]
            self.__init_model(img for img in images_pillow)
            return self.__predict_photos(images_pillow, boxOrPoint)
        else:
            images_pillow = [load_image_from_path(image_path)[0] for image_path in images]
            self.__init_model(img for img in images_pillow)
            return self.__predict_photos(images_pillow, boxOrPoint)

    def __predict_photos(self, images, boxesOrPoints: List[BoxOrPoint]):
        def all_elements_are_not_null(arr, func):
            return all([func(a) is not None for a in arr])

        mask_response = {}
        with torch.inference_mode(), torch.autocast(self.device.type):
            for frame_idx in tqdm(range(len(images)), desc="Predicting photos"):
                self.sam2_model.set_image(images[frame_idx])
                points_per_frame = sorted([bpl for bpl in boxesOrPoints if bpl.frame == frame_idx],
                                          key=lambda item: item.object_id)
                object_ids = [bpl.object_id for bpl in points_per_frame]
                # masks, scores, logits
                np_boxes = np.array(
                    [bpl.bbox for bpl in points_per_frame]) if all_elements_are_not_null(points_per_frame,
                                                                                         lambda x: x.bbox) else None
                np_point = np.array(
                    [bpl.point for bpl in points_per_frame]) if all_elements_are_not_null(points_per_frame,
                                                                                          lambda x: x.point) else None
                mask_logits, scores, logits = self.sam2_model.predict(box=np_boxes if np_boxes is not None else None,
                                                                      point_labels=np.array([bpl.label for bpl in
                                                                                             points_per_frame]) if np_point is None else None,
                                                                      point_coords=np_point if np_point is not None else None,
                                                                      return_logits=False,
                                                                      multimask_output=False)
                masks = (mask_logits > 0.0).astype(bool)
                masks = masks.squeeze(axis=1)
                mask_response[frame_idx] = []
                for object_id in object_ids:
                    mask_per_object = masks[object_id, :, :]
                    true_values = [(i, j) for i in range(mask_per_object.shape[0]) for j in
                                   range(mask_per_object.shape[1]) if mask_per_object[i, j]]
                    mask_response[frame_idx].append(
                        MaskResponse(
                            image_shape=mask_per_object.shape[::-1],
                            true_values=true_values,
                            object_id=object_id
                        ))
        return mask_response


class SAM2Serve:
    loaded_models = {}

    def get_or_load_model(self, model_name, **kwargs):
        model = self.loaded_models.get(model_name)
        if not model:
            self.loaded_models[model_name] = SAM2(model_name, **kwargs)
            return self.loaded_models[model_name]
        else:
            return model
