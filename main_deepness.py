import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

def preprocess_image(image, input_size):
    # Resize and normalize the image
    resized = cv2.resize(image, input_size)
    input_image = resized.astype(np.float32) / 255.0
    # Transpose to channel-first format
    input_image = input_image.transpose(2, 0, 1)
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def postprocess_output(output, conf_threshold=0.25, iou_threshold=0.45):
    # Assuming the output format matches YOLOv7
    # You may need to adjust this based on the actual output format of your ONNX model
    boxes = output[0]
    scores = output[1]
    class_ids = output[2]

    # Filter detections based on confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices]

def main():
    # Load the ONNX model
    model_path = "C:/Users/david.bruner/Documents/DBRepos/CarsAerial/models/car_aerial_detection_yolo7_ITCVD_deepness.onnx"
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Get model info
    inputs = session.get_inputs()
    input_shape = inputs[0].shape
    input_name = inputs[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Path to your image
    image_path = "auckland-0075m-urban-aerial-photos-2017.jpg"

    # Read and preprocess the image
    image = cv2.imread(image_path)
    input_image = preprocess_image(image, (input_shape[2], input_shape[3]))

    # Perform inference
    outputs = session.run(output_names, {input_name: input_image})

    print(outputs)
    # Postprocess the output
    boxes, scores, class_ids = postprocess_output(outputs)

    # Create Detections object
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids.astype(int)
    )

    # Filter detections for cars (assuming car class_id is 2, adjust if necessary)
    # mask = np.isin(detections.class_id, [2])
    # car_detections = detections[mask]

    # Initialize annotator
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate image
    annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Save the result
    cv2.imwrite("result.jpg", annotated_image)

    print(f"Detection complete. Result saved as 'result.jpg'")
    print(f"Number of detections: {len(detections)}")

if __name__ == "__main__":
    main()