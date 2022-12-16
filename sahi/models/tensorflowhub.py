import logging
from typing import List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import  check_requirements

logger = logging.getLogger(__name__)

class TensorflowHubDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["tensorflow", "tensorflow_hub"])

    def set_device(self):
        import tensorflow as tf

        if not (self.device):
            self.device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"

    def load_model(self):
        import tensorflow as tf
        import tensorflow_hub as hub

        if "tfhub.dev/tensorflow" in self.model_path:
            with tf.device(self.device):
                self.model = hub.load(self.model_path)
        else:
            raise ValueError(
                "Check 'https://tfhub.dev/tensorflow/collections/object_detection/' for supported TF Hub models."
            )

        if self.category_mapping is None:
            from sahi.utils.tensorflow import COCO_CLASSES

            category_mapping = {str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        from sahi.utils.tensorflow import resize, to_float_tensor

        if self.image_size is not None:
            img = to_float_tensor(image)
            img = resize(img, self.image_size)
            prediction_result = self.model(img)

        else:
            img = to_float_tensor(image)
            prediction_result = self.model(img)

        self._original_predictions = prediction_result
        # TODO: add support for multiple image prediction
        self.image_height, self.image_width = image.shape[0], image.shape[1]

    @property
    def num_categories(self):
        num_categories = len(self.category_mapping)
        return num_categories

    @property
    def has_mask(self):
        # TODO: check if model output contains segmentation mask
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        import tensorflow as tf

        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        boxes = original_predictions["detection_boxes"][0].numpy()
        scores = original_predictions["detection_scores"][0].numpy()
        category_ids = original_predictions["detection_classes"][0].numpy()

        # create object_prediction_list
        object_prediction_list = []
        object_prediction_list_per_image = []
        with tf.device(self.device):
            for i in range(min(boxes.shape[0], 100)):
                if scores[i] >= self.confidence_threshold:
                    score = float(scores[i])
                    category_id = int(category_ids[i])
                    category_names = self.category_mapping[str(category_id)]
                    box = [float(box) for box in boxes[i]]
                    x1, y1, x2, y2 = (
                        int(box[1] * self.image_width),
                        int(box[0] * self.image_height),
                        int(box[3] * self.image_width),
                        int(box[2] * self.image_height),
                    )
                    bbox = [x1, y1, x2, y2]

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        bool_mask=None,
                        category_id=category_id,
                        category_name=category_names,
                        shift_amount=shift_amount,
                        score=score,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
            self._object_prediction_list_per_image = object_prediction_list_per_image