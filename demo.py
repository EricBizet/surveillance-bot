import cv2
import numpy as np
import onnxruntime

image = cv2.imread("beatles-abbeyroad.jpg")
image = cv2.resize(image, (640, 640))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession("yolo_nas_s.onnx", providers=["CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

print(f"{result[0].shape} - {result[1].shape} - {result[2].shape} - {result[3].shape}")
