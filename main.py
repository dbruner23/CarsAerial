import cv2
from ultralytics import YOLO
import supervision as sv
import torch
import numpy as np

def main():
    # Load the YOLOv9 model
    model = YOLO("yolov9c.pt")  # or the specific path to your YOLOv9 weights
    if torch.cuda.is_available():
        print("Using GPU")
        model.to("cuda")

    # Path to your image
    image_path = "image_data/albany-mid-zoom.jpg"

    # Read the image
    image = cv2.imread(image_path)

    # Perform inference
    results = model(image)[0]

    # Initialize detections
    detections = sv.Detections.from_ultralytics(results)
    mask = np.isin(detections.class_id, [0,1,2,3,4,5,6,7,8])
    # Filter detections for cars (assuming car class_id is 2, adjust if necessary)
    car_detections = detections[mask]

    # Initialize annotator
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate image
    annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=car_detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=car_detections)

    # Save the result
    cv2.imwrite("image_data/result.jpg", annotated_image)

    print(f"Detection complete. Result saved as 'result.jpg'")
    print(f"Number of detected cars: {len(car_detections)}")

if __name__ == "__main__":
    main()