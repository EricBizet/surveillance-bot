from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion import DetectionOutputFormatMode

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

export_result = model.export(
    "yolo_nas_s.onnx",
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    confidence_threshold = 0.6,
    nms_threshold = 0.5,
    num_pre_nms_predictions = 100,
    max_predictions_per_image = 5
)