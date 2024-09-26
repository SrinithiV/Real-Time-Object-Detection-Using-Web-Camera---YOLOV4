# Real Time Object Detection Using Web Camera YOLO v4
## AIM :
To Perform real-time object detection using a trained YOLO v4 model through laptop camera.

## ALGORITHM :
1.Load YOLOv4 network.

2.Load the COCO class labels.

3.Set up video capture for webcam.

4.Prepare the image for YOLOv4.

5.Get YOLO output.

6.Initialize lists to store detected boxes, confidences, and class IDs.

7.To detect the object.

8.Calculate top-left corner of the box.

9.Apply Non-Max Suppression to eliminate redundant overlapping boxes.

10.Draw bounding boxes and labels on the image.

11.Green color for bounding boxes.

12.Show the image with detected objects.

13.Release video capture and close windows.

## PROGRAM :
DEVELOPED BY : SRINITHI V

REGISTER NO : 212222110046
```py
import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)  
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
## OUTPUT :
![Screenshot 2024-09-26 112402](https://github.com/user-attachments/assets/ba9753a3-44ac-4642-95db-c5171534e965)

## RESULT :
Thus , the real-time object detection using a trained YOLO v4 model through laptop camera has been perfomed successfully.


