import uvicorn
from fastapi import FastAPI , UploadFile
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import io
from PIL import Image
import cv2
import base64

'''FastAPI Script Description for Emotion Detection
This FastAPI script is designed for emotion detection from images. It uses a deep learning model to identify and classify emotions on human faces in the uploaded images. The script can be used as a backend for a web or mobile application.

Main Features:
REST API: The script implements a REST API using FastAPI, a modern and fast framework for creating APIs with Python.
Model Loading: The deep learning model is loaded from a JSON file (youssef.json) for its structure and an H5 file (youssef.h5) for its weights. This model is used for predicting emotions.
Image Preprocessing: Images sent via the API are preprocessed to be compatible with the model's requirements. This includes converting the images into a format compatible with OpenCV, detecting faces, and preparing ROIs (Regions Of Interest) for prediction.
Emotion Detection: Faces detected in the images are analyzed by the model to identify emotions. The supported emotions are anger (Angry), happiness (Happy), neutral (Neutral), and sadness (Sad).
Returning Results: The API returns the detected emotions for each face in the image, along with the modified image showing visual indications of the detected emotions.

API Endpoints:
GET /: A simple endpoint to test if the API is operational. Returns a welcome message.
POST /predect: The main endpoint for uploading images and receiving emotion predictions.

Usage:
To use the API, send an image via a POST request to the /predect endpoint. The API will process the image, detect faces, and return the identified emotions for each detected face, as well as the modified image.'''


app = FastAPI ()

@app.get("/")
async def greet():
    # Returns a simple greeting message
    return {"message": "Bonjour"}

# Load the model structure from a JSON file
emotion_model = model_from_json(open("youssef.json", "r").read())

# Load the model weights
emotion_model.load_weights('youssef.h5')

# Dictionary to map indices to emotions
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}


# Function for image preprocessing
def preprocess(image_stream):
    # Convert a PIL image to an OpenCV image (numpy array)
    frame = np.array(image_stream)

    # Get the dimensions of the image
    height, width, _ = frame.shape

    # Convert the image to BGR format (used by OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_copy = frame.copy()
    
    # Detect faces using CascadeClassifier
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)
    
    # Initialize lists to store results
    face_list = []
    pred_list = []
    full_pred_list = []
    roi_coordinates = [] # Coordinates of detected faces
    
    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Convert the ROI (Region Of Interest) to grayscale
        roi_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h+10), (255,255,0), 2)
        
        # Prepare the image for prediction (resize and normalize)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        
        # Get the index of the emotion with the highest probability
        maxindex = int(np.argmax(emotion_prediction))
        
        # Display the predicted emotion
        cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        # Store the coordinates of the ROI
        roi_coordinates.append((x, y, x + w, y + h))
        
        # Add the face and predicted emotion to the lists
        pred_list.append(maxindex)
        full_pred_list.append(emotion_prediction)
        
    # Extract faces from the original image
    for (x, y, x1, y1) in roi_coordinates:
        face_list.append(frame_copy[y:y1, x:x1])
        
    # Resize the image with rectangles and texts
    full_img = cv2.resize(frame,(width, height),interpolation = cv2.INTER_CUBIC)
    full_img = np.expand_dims(full_img, 0)  # Ajout des dimensions de batch et de canal
    
    return face_list, pred_list, full_img, full_pred_list

@app.post("/predect")
async def predect(file:UploadFile):
    # Read the image sent by the user
    image_data = await file.read()
    image_stream = Image.open(io.BytesIO(image_data))
    image_stream.seek(0)

    # Preprocess the image using the preprocess function
    img_processed = preprocess(image_stream)
    
    # Retrieve the predictions
    predictions = img_processed[1]
    
    img_base64_list = []
    
    # Convert images to base64 to send back to the client
    for i in range(len(img_processed[0])):
        _, img_encoded = cv2.imencode('.jpg', img_processed[0][i])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        img_base64_list.append(img_base64)
    
    # Convert the full image to base64 to send back to the client
    _, full_img_encoded = cv2.imencode('.jpg', img_processed[2][0])
    full_img_base64 = base64.b64encode(full_img_encoded.tobytes()).decode('utf-8')
    
    pred = [None] * len(predictions)
    # Convert predictions to emotions
    for i in range(len(predictions)):
        pred[i] = emotion_dict[predictions[i]]
    
    full_pred = img_processed[3]

    return {'prediction':pred, 'image': img_base64_list, 'full_image': full_img_base64, 'full_prediction': [prediction.tolist() for prediction in full_pred]}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)