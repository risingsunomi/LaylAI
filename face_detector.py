# Using YOLO, detect faces in a image

import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, confidence=0.5, threshold=0.3):
        self.config_path = "{}/lib/yolov3.cfg".format(os.getcwd())
        self.weights_path = "{}/lib/yolov3.weights".format(os.getcwd())
        self.classes_path = "{}/lib/coco.names".format(os.getcwd())

        self.confidence = confidence
        self.threshold = threshold

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        
        self.classes = None
        with open(self.classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        
    def detect_faces(self, image):
        (H, W) = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )
        
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.get_output_layers())
        
        boxes = []
        confidences = []
        class_ids = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # filter to just person detection
                if confidence > self.confidence and self.classes[class_id] == "person":
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        faces = []
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            faces.append((x, y, x+w, y+h))
        
        return faces
    
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def add_boxes_to_faces(image, faces):
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image
