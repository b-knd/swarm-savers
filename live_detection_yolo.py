import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")
print("YOLO model loaded successfully")

# Get layer names
layer_names = net.getLayerNames()
print("Layer names:", layer_names)

# Get output layer indices
output_layer_indices = net.getUnconnectedOutLayers()

# Convert output layer indices to layer names
output_layers = []
for idx in output_layer_indices.flatten():
    layer_name = layer_names[idx - 1] if idx > 0 else ""
    output_layers.append(layer_name)

print("Output layers:", output_layers)

# Load class names
classes = []
with open("darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video feed
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Video/Video1.mov")

# Set the minimum confidence threshold for detection
min_confidence = 0.5

# Set the Non-maximum Suppression (NMS) threshold
nms_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store bounding boxes and confidence scores
    bounding_boxes = []
    confidence_scores = []

    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence and class_id == 0:  # 0 corresponds to 'person' class in COCO dataset
                # Survivor detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bounding_boxes.append([x, y, w, h])
                confidence_scores.append(float(confidence))

    # Apply NMS to filter out redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_scores, min_confidence, nms_threshold)

    # Showing information on the screen
    if len(indices) > 0:
        for i in indices.flatten():
            box = bounding_boxes[i]
            x, y, w, h = box
            confidence = confidence_scores[i]

            # Draw rectangle around the survivor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text with survivor and confidence score
            text = f'Survivor: {confidence:.2f}'
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with survivor detection
        cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
