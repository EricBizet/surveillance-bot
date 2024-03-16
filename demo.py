import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization

def show_predictions_from_flat_format(image, predictions):
    [flat_predictions] = predictions

    image = image.copy()
    class_names = COCO_DETECTION_CLASSES_LIST
    DetectionVisualization._generate_color_mapping(len(class_names))

    class_names=COCO_DETECTION_CLASSES_LIST

    first_batch = flat_predictions[flat_predictions[:, 0] == 0] # [N, 7] with first index representing the batch index
    predictions = first_batch[:, 1:] # [N, 6]


    image = DetectionVisualization.visualize_image(
        image_np=image,
        class_names=COCO_DETECTION_CLASSES_LIST,
        pred_boxes=predictions
    )


    cv2.imwrite("output.jpg", image)

image = cv2.imread("beatles-abbeyroad.jpg")
image = cv2.resize(image, (640, 640))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession("yolo_nas_s_int8.onnx", providers=["CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

print(f"{result[0].shape}")
show_predictions_from_flat_format(image, result)