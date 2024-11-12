from itertools import groupby
from typing import Union, List

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from api.patches import DEVICE
from api.utils import is_base64_string, base64_to_image_with_size, load_image_from_path, \
    offload_video_as_images
from models import BoxOrPoint, MaskResponse, PredictResponse


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
        elif not self.sam2_model and images:
            self.sam2_model = SAM2ImagePredictor.from_pretrained(self.model_name, device=self.device.type,
                                                                 token=self.hf_token)
        if images_path:
            if self.inference_state is not None:
                self.sam2_model.reset_state(self.inference_state)

            self.inference_state = self.sam2_model.init_state(video_path=images_path,
                                                              offload_video_to_cpu=True,
                                                              offload_state_to_cpu=True
                                                              )

    def call_model(self, images, video, box_or_point, scale_factor,
                   start_second, end_second, stream=False):
        if video is not None:
            images_path = offload_video_as_images(video, scale_factor, start_second, end_second)
            self.__init_model(images_path=images_path)
            result = self.__predict_video(box_or_point, stream)
        elif images is not None and is_base64_string(images[0]):
            images_pillow = [base64_to_image_with_size(image)[0] for image in images]
            self.__init_model(img for img in images_pillow)
            result = self.__predict_photos(images_pillow, box_or_point, stream)
        else:
            images_pillow = [load_image_from_path(image_path)[0] for image_path in images]
            self.__init_model(img for img in images_pillow)
            result = self.__predict_photos(images_pillow, box_or_point, stream)
        return result

    def __predict_photos(self, images, box_or_point: List[BoxOrPoint], stream=False):
        grouped_items = self.__group_box_point_by_frame_obj_id(box_or_point, only_frame=True)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if stream:
                for frame_idx, boxes in tqdm(grouped_items.items(), desc="Predicting photos"):
                    mask_responses = self.__predict_mask_responses(image=images[frame_idx],
                                                                   box_or_point_by_frame=boxes)
                    yield PredictResponse(response={frame_idx: mask_responses}).model_dump_json() + "\n"
            else:
                mask_response = {}
                for frame_idx, boxes in tqdm(grouped_items.items(), desc="Predicting photos"):
                    mask_responses = self.__predict_mask_responses(image=images[frame_idx],
                                                                   box_or_point_by_frame=boxes)
                    mask_response[frame_idx] = mask_responses
                return PredictResponse(response=mask_response)

    def __predict_mask_responses(self, image, box_or_point_by_frame):
        self.sam2_model.set_image(image)
        points_per_frame = sorted(box_or_point_by_frame,
                                  key=lambda item: item.object_id)
        object_ids = [bpl.object_id for bpl in points_per_frame]
        np_boxes = np.array(
            [bpl.bbox for bpl in points_per_frame]) if self.__all_elements_are_not_null(points_per_frame,
                                                                                        lambda
                                                                                            x: x.bbox) else None
        np_point = np.array(
            [bpl.point for bpl in points_per_frame]) if self.__all_elements_are_not_null(points_per_frame,
                                                                                         lambda
                                                                                             x: x.point) else None
        mask_logits, scores, logits = self.sam2_model.predict(box=np_boxes if np_boxes is not None else None,
                                                              point_labels=np.array([bpl.label for bpl in
                                                                                     points_per_frame]) if np_point is None else None,
                                                              point_coords=np_point if np_point is not None else None,
                                                              return_logits=False,
                                                              multimask_output=False)
        masks = (mask_logits > 0.0).astype(bool)
        masks = masks.squeeze(axis=1)
        return [self.parse_to_model(object_id, masks) for object_id in object_ids]

    def __predict_video(self, box_or_point: List[BoxOrPoint], stream=False):
        grouped_items = self.__group_box_point_by_frame_obj_id(box_or_point)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for frame_idx, boxes_by_obj_id in grouped_items.items():
                for object_id, boxes in boxes_by_obj_id.items():
                    self.__add_points_or_boxes(frame_idx, object_id, boxes)

            mask_response = {}
            for frame_idx, object_ids, mask_logits in self.sam2_model.propagate_in_video(
                    inference_state=self.inference_state):
                masks = torch.squeeze(mask_logits, dim=1)
                # Prepare the response for this frame
                frame_response = {
                    frame_idx: [
                        self.parse_to_model(object_ids[idx], (masks[idx].cpu().numpy() > 0.0).astype(bool))
                        for idx in range(len(object_ids))
                    ]
                }
                if stream:
                    yield PredictResponse(response=frame_response).model_dump_json() + '\n'
                else:
                    mask_response.update(frame_response)
            return mask_response

    @staticmethod
    def __group_box_point_by_frame_obj_id(box_or_point: List[BoxOrPoint], only_frame=False):
        items_sorted = sorted(box_or_point, key=lambda x: x.frame)
        if only_frame:
            return groupby(items_sorted, key=lambda x: x.frame)
        else:
            grouped_items = {
                frame: {
                    object_id: list(bboxes)[0]  # we except only one box for each object in the frame
                    for object_id, bboxes in groupby(boxes, key=lambda box: box.object_id)
                }
                for frame, boxes in groupby(items_sorted, key=lambda x: x.frame)
            }
            return grouped_items

    @staticmethod
    def parse_to_model(object_id, masks):
        if len(masks.shape) >= 3:
            mask_per_object = masks[object_id, :, :]
        else:
            mask_per_object = masks
        true_values = [(i, j) for i in range(mask_per_object.shape[0]) for j in
                       range(mask_per_object.shape[1]) if mask_per_object[i, j]]
        return MaskResponse(
            image_shape=mask_per_object.shape[::-1],
            true_values=true_values,
            object_id=object_id
        )

    def __add_points_or_boxes(self, frame_idx, object_id, boxes):
        if boxes.bbox is not None:
            np_box = np.array(boxes.bbox)
            _, obj_id_res, video_res_masks = self.sam2_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=np_box
            )
        else:
            np_point = np.array(boxes.point)
            np_label = np.array(boxes.label)
            _, obj_id_res, video_res_masks = self.sam2_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=np_point,
                labels=np_label
            )

    @staticmethod
    def __all_elements_are_not_null(arr, func):
        return all([func(a) is not None for a in arr])


class SAM2Serve:
    loaded_models = {}

    def get_or_load_model(self, model_name, **kwargs):
        model = self.loaded_models.get(model_name)
        if not model:
            self.loaded_models[model_name] = SAM2(model_name, **kwargs)
            return self.loaded_models[model_name]
        else:
            return model
