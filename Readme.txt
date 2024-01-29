Emotion Detection Application
This application is a combination of a Streamlit frontend and a FastAPI backend designed to detect emotions from images. It uses a machine learning model to analyze faces in images and identify emotions such as anger, happiness, neutrality, and sadness.

Features
Emotion Detection: Detects four main emotions: Angry, Happy, Neutral, Sad.
User-Friendly Interface: Utilizes Streamlit for an intuitive and interactive user interface.
Real-Time Image Processing: Ability to upload images or use a webcam to capture and analyze emotions instantly.
REST API: Incorporates a FastAPI backend for efficient processing and response.
How to Use
Start the FastAPI Server:

Ensure FastAPI and other dependencies are installed.
Run the FastAPI server by executing uvicorn main:app --reload.
Launch the Streamlit Application:

Open another terminal.
Start the Streamlit app by running streamlit run app.py.
Interacting with the Application:

Navigate through the Streamlit interface.
Upload an image or use the webcam feature to detect emotions.
View the detected emotions displayed alongside the processed image.