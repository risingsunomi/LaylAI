#!/usr/bin/env python3

# Gets an emotion from a face image using a trained tensorflow model

import cv2
import tensorflow as tf
import os
from face_detector import FaceDetector

class FaceEmotion:
    def __init__(self):
        # Load YOLO face detector
        self.face_detector = FaceDetector()

        # Load the TensorFlow model for emotion recognition
        self.model = tf.keras.models.load_model("model/[FERENS]face_recognition_model__20230217_060809.h5")

        # Define a dictionary to map emotions to labels
        self.emotion_dict = {
            0: "Angry", 
            1: "Contempt", 
            2: "Disgust", 
            3: "Fear", 
            4: "Happy", 
            5: "Neutral", 
            6: "Sad",
            7: "Surprise"}

        # Read the video stream
        self.cap = cv2.VideoCapture(0)

    def capture_emotions(self):
        while True:
            # Read a frame from the video stream
            ret, frame = self.cap.read()
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = self.face_detector.detect_faces(frame)
            
            # Loop over the faces and predict their emotions
            for (x, y, w, h) in faces:
                # Extract the face from the frame
                face = gray[y:y+h, x:x+w]
                
                # Resize the face to (48, 48) for the TensorFlow model
                face = cv2.resize(face, (48, 48))
                
                # Expand the face to (1, 48, 48, 1) for TensorFlow
                face = tf.expand_dims(face, axis=-1)
                face = tf.expand_dims(face, axis=0)
                
                # Predict the emotion using the TensorFlow model
                prediction = self.model.predict(face)
                
                # Get the index of the highest confidence emotion
                emotion = self.emotion_dict[tf.argmax(prediction[0])]
                
                # Draw a rectangle around the face and label the emotion
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            
            # Show the frame
            cv2.imshow("Emotion Recognition", frame)
            
            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Release the video stream
            self.cap.release()

            # Close the window
            cv2.destroyAllWindows()


if __name__ == "__main__":
    fe = FaceEmotion()
    fe.capture_emotions()
