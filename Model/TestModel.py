import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}

folder_names = ['Angry', 'Happy', 'Neutral', 'Sad']

keyword = 'Model_256'

# Create the directory if it doesn't exist
if not os.path.exists(f'results_{keyword}'):
    os.makedirs(f'results_{keyword}')
    
# Load the model structure from json file
emotion_model = model_from_json(open(f"results_{keyword}/model_{keyword}.json", "r").read())

# Load the model weights
emotion_model.load_weights(f'results_{keyword}/emotion_model_{keyword}.h5')
print("Loaded model from disk")

true_labels = []
predicted_labels = []

# Add the path to the folder containing the four folders
base_path = 

# Initialize lists to store the data
photo_names = []
true_labels = []
predicted_labels = []
matches = []

for folder_name in folder_names:
    # Iterate over the images in the folder
    for filename in os.listdir(os.path.join(base_path, folder_name)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load and preprocess the image
            image = cv2.imread(os.path.join(base_path, folder_name, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (48, 48))
            image = np.reshape(image, (1, 48, 48, 1))
            image = image / 255.0

            # Make predictions using the loaded model
            predictions = emotion_model.predict(image)
            emotion_label = emotion_dict[np.argmax(predictions)]

            # Compare the predicted emotion label with the folder name
            if emotion_label == folder_name:
                print(f"Image: {filename}, Emotion: {emotion_label}, Folder: {folder_name}")
            
            photo_names.append(filename)
            true_labels.append(folder_name)
            predicted_labels.append(emotion_label)
            matches.append(emotion_label == folder_name)

# Generate classification report and confusion matrix
print("Classification Report:")
report = classification_report(true_labels, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f'results_{keyword}/classification_report_{keyword}.csv')
print(report_df)

print("Confusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
# Use seaborn to plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_dict.values(), yticklabels=emotion_dict.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f'results_{keyword}/confusion_matrix_{keyword}.png')

# Create a DataFrame from the lists
df = pd.DataFrame({
    'photo': photo_names,
    'true': true_labels,
    'pred': predicted_labels,
    'match': matches
})

# Write the DataFrame to a CSV file
df.to_csv('predictions.csv', index=False)