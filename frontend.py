import streamlit as st
import requests
from PIL import Image
import io
from streamlit_option_menu import option_menu
import base64

# Constants configuration
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
emotion_emote_dict = {
    "Angry": "😡",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢"
}

FASTAPI_SERVER_URL = "http://localhost:4000/predect"

def page_config():
    # Configures page settings for Streamlit
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon=":smiley:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .image-carousel {
            max-width: 100%;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    
    with st.sidebar:choix = option_menu("Menu", ["Contexte","Téléchargez une image", "Mets ta cam"],
                                        icons=['house','upload','camera fill']
    )
    return choix

# Displays the project context
def context():
    # Displaying the project context
    st.markdown("<p class='big-font'>Contexte du Projet:</p>", unsafe_allow_html=True)


    st.markdown("""
    <br>ProjetFinal_JedhaDS27_Emotions - Emotion detector on photos and videos for emotion sensibility disorders<br>


    Les enfants autistes sont connus pour identifier difficilement leurs émotions dans
    les situations sociales.la joie est ressentie plus intensément que les émotions négatives
    (tristesse, colère) et la peur n'est pas ressentie avec une grande intensité.


    Toutes les études concernant la reconnaissance des émotions dans l'autisme de 1989 à 2011 confirme la reconnaissance du
    bonheur chez les personnes autistes, avec un taux de reconnaissance de 95 %, ainsi que la
    non-reconnaissance de la peur, dont le taux est proche de 0. Les autres émotions négatives
    (tristesse, colère et dégoût) sont plus faiblement reconnues que le bonheur mais mieux
    reconnues que la peur (Uljarevic et Hamilton, 2013).
    Aussi, les autistes ressentent moins d'émotions complexes, comme l'embarras, la
    honte ou la fierté (Capps, Yirmiya & Sigman, 1992). Ces émotions sont aussi dites
    « centrées sur le soi » ou, parfois, appelées « émotions sociales »
                    """, unsafe_allow_html=True)


# Function to display the title of the application
def display_title():
    st.title("Reconnaissance des émotions")
    st.write("Téléchargez une image et le modèle prédira l'émotion")
    
# Function to handle the upload of images
def upload():
    # File upload widget in Streamlit
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])
    buffer = None
    
    # Create columns to display the uploaded image and the prediction
    col_empty, col1, col2 = st.columns([1,1,2])
            
    # When a file is uploaded
    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Define the maximum size
        max_size = (1000, 1000)
        
        # Resize the image
        image.thumbnail(max_size)

        col1.image(image, caption='Image téléversée', use_column_width=False)
        
        # Convert the image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)   
        
    return uploaded_file, buffer

def emo_pred(uploaded_file, buffer):
    
    try:
        # Send request to FastAPI server
        files = {"file": (uploaded_file.name, buffer, "multipart/form-data")}
        response = requests.post(FASTAPI_SERVER_URL, files=files)
        return response
        
    except Exception as e:
        st.write("Erreur lors de la requête POST au serveur FastAPI : " + str(e))
        return None
        
def display_full_image(response_data): 
    # Check if the response contains the 'full_image' key
    if "full_image" in response_data:
        # Decode the base64 string into bytes
        full_image = decode_image(response_data["full_image"])
        # Create columns to display the uploaded image and the prediction
        col_empty, col1, col2 = st.columns([1,1,2])
        # Display the processed image
        col1.image(full_image, caption='Image traitée', use_column_width=False)

def display_predictions(response_data):
    # Check if the response contains the 'image' and 'prediction' keys
    if "image" in response_data:
        image_list = response_data["image"] # List of images
        prediction = response_data["prediction"] # List of predictions
        full_prediction = response_data["full_prediction"] # List of full predictions
        
        # Decode the base64 string into bytes
        for i in range(len(image_list)):
            image = decode_image(image_list[i])
            pred = prediction[i]
            
            # Create columns to display the uploaded image and the prediction
            col_empty, col1, col2 = st.columns([1,1,2])
            
            emotion_emote = emotion_emote_dict[pred]
            # Display the image without a caption
            col1.image(image, width=200)

            # Display the caption using markdown with a larger font size
            col1.markdown(f"<h2>Émotion prédite : {pred} {emotion_emote}</h2>", unsafe_allow_html=True)
            
            # Display the prediction using a progress bar
            progress_value = [None] * len(full_prediction[i][0])
            
            for j, score in enumerate(full_prediction[i][0]):
                emotion = emotion_dict[j]
                col2.markdown(f"**Probabilité d'émotion: {emotion} : {score*100:.2f}%**")
                progress_value = int(score * 100)
                col2.progress(progress_value)
                
            # Display a horizontal line between images
            if i < len(image_list) - 1:  # Don't display a line after the last image
                st.markdown("---")    
    else:
        st.write("Aucune donnée d'image trouvée dans la réponse")

def cam():
    # File uploader
    uploaded_file = st.camera_input("Allume ta webcam et prends toi directement en photo!")
    response = None
    
    try:
        if uploaded_file is not None:
            # Process the image here (e.g., save it to a specific location)
            st.success("Ton image a bien été téléchargée!")
            st.write("Ta prédiction va bientôt apparaître!")
            # Send request to FastAPI server
            api_url = FASTAPI_SERVER_URL
            data = {"file": uploaded_file.getvalue(), "type":"image/jpeg"}
            response = requests.post(api_url, files=data)
            return response
            
    except Exception as e:
        st.write("Erreur lors de la requête POST au serveur FastAPI : " + str(e))
        return None
        

def decode_image(image_data):
    # Decode the base64 string into bytes
    image_data = base64.b64decode(image_data)
    # Convert the bytes to an image
    image = Image.open(io.BytesIO(image_data))
    return image


def main():
    choix = page_config()
    
    if choix == "Contexte":
        context()

    elif choix == "Téléchargez une image":
        display_title()
        uploaded_file, buffer = upload()
        if st.button('Prédire'):
            response = emo_pred(uploaded_file, buffer)
            if response.status_code == 200:
                response_data = response.json()
                display_full_image(response_data)
                display_predictions(response_data)
                
    elif choix == "Mets ta cam" :
        response = cam()
        if response is not None:
            if response.status_code == 200:
                response_data = response.json()
                display_full_image(response_data)
                display_predictions(response_data)
            else:
                st.write("Une erreur s'est produite, code d'état :", response.status_code)

# Run the app
if __name__ == "__main__":
    main()