from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("./models/yolov8/best_5.pt")

# Load and detect objects in an image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\new_engine_chassis_numbers\cropped\7.jpeg"  # Change to your test image
resized = cv2.resize(cv2.imread(image_path), (768, 768))
results = model(resized)

# Show results
for result in results:
    boxes = result.boxes.xyxy  # Bounding boxes
    labels = result.boxes.cls  # Class labels
    confs = result.boxes.conf  # Confidence scores

    img = cv2.imread(image_path)
    for box, label, conf in zip(boxes, labels, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{model.names[int(label)]} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
