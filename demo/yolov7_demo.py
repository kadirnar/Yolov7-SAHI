from sahi.utils.yolov7 import download_yolov7_model
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi import AutoDetectionModel

# download yolov7 model
yolov7_model_path = 'models/yolov7.pt'
download_yolov7_model(yolov7_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7pip', # or 'yolov7hub'
    model_path=yolov7_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="demo_data/")
