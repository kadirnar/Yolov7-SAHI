from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi import AutoDetectionModel

# set tensorflow efficientdet model 
model_path = "https://tfhub.dev/tensorflow/efficientdet/d0/1"

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')


detection_model = AutoDetectionModel.from_pretrained(
    model_type='tensorflowhub', 
    model_path=model_path,
    confidence_threshold=0.3,
    device="GPU", # or 'cpu'
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








