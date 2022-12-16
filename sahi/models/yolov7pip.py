import logging
from typing import List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import  check_requirements

logger = logging.getLogger(__name__)

class Yolov7PipDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["torch", "yolov7detect"])

    def load_model(self):
        import yolov7

        try:
            model = yolov7.load(self.model_path, device=self.device)
            model.conf = self.confidence_threshold
            self.model = model
        except ImportError:
            raise ImportError(
                "Please run 'pip install yolov7detect' to install Yolov7 first for Yolov7 inference."
            )
        except Exception as e:
            TypeError("model_path is not a valid torchvision model path: ", e)

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        check_requirements(["torch"])
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        if self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list_per_image = []
        object_prediction_list = []
        for _, image_predictions_in_xyxy_format in enumerate(self._original_predictions.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                category_id = pred[5]
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=int(category_id),
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image