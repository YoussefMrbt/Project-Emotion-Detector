import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report
import os
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('Agg')

class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        is_correct = tf.equal(y_pred, y_true)
        self.correct.assign_add(tf.reduce_sum(tf.cast(is_correct, tf.float32)))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.)
        self.total.assign(0.)

keyword = 'model_256'

# Create the directory if it doesn't exist
if not os.path.exists(f'results_{keyword}'):
    os.makedirs(f'results_{keyword}')
    
#intialize image data generator with Rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3)

test_datagen = ImageDataGenerator(
    rescale=1./255)

#Preprocess the training data
train_generator = train_datagen.flow_from_directory(
    'data/trainv2',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='training'  # set as training data
)
# Calculate the class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)

# Convert class weights to dictionary
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)


#Preprocess the validation data
validation_generator = validation_datagen.flow_from_directory(
    'data/trainv2',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='validation'  # set as validation data
)

# Load and preprocess the test data
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

#Create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.5))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(256, activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(4, activation='softmax'))


# Define the early stopping callback
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1)



#Define the learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=False)

#Define the optimizer
optimizer = Adam(learning_rate=lr_schedule)

#Compile the model
emotion_model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer, 
                    metrics=[CustomAccuracy(), F1Score(num_classes=4)])

# Create a checkpoint callback
checkpoint = ModelCheckpoint(f'results_{keyword}/weights.' + '{epoch:02d}-{val_custom_accuracy:.2f}.hdf5', monitor='val_custom_accuracy', verbose=1, save_best_only=True, mode='max')

#Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//64,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//64,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint]
)
# Test the model on the training data
train_loss, train_custom_accuracy, train_f1 = emotion_model.evaluate(train_generator)
print('Train loss:', train_loss)
print('Train accuracy:', train_custom_accuracy)
print('Train F1 score:', train_f1)


# Test the model on the validation data
val_loss, val_custom_accuracy, val_f1 = emotion_model.evaluate(validation_generator)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_custom_accuracy)
print('Validation F1 score:', val_f1)


# Test the model on the test data
test_loss, test_custom_accuracy, test_f1 = emotion_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_custom_accuracy)
print('Test F1 score:', test_f1)

# Export the results to a file
with open(f'results_{keyword}.txt', 'w') as f:
    f.write(f'Train loss: {train_loss}\n')
    f.write(f'Train accuracy: {train_custom_accuracy}\n')
    f.write(f'Train F1 score: {train_f1}\n')
    f.write(f'Validation loss: {val_loss}\n')
    f.write(f'Validation accuracy: {val_custom_accuracy}\n')
    f.write(f'Validation F1 score: {val_f1}\n')
    f.write(f'Test loss: {test_loss}\n')
    f.write(f'Test accuracy: {test_custom_accuracy}\n')
    f.write(f'Test F1 score: {test_f1}\n')

# Compute the confusion matrix for the training set
confusion_mtx_train = tf.math.confusion_matrix(train_generator.classes, np.argmax(emotion_model.predict(train_generator), axis=1))

# Convert the confusion matrix to numpy array
confusion_mtx_train = confusion_mtx_train.numpy()

# Plot the confusion matrix for the training set
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_train, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Training Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save the plot as an image file
plt.savefig(f'results_{keyword}\confusion_matrix__{keyword}_train.png')
plt.close()

# Compute the confusion matrix for the validation set
confusion_mtx_val = tf.math.confusion_matrix(validation_generator.classes, np.argmax(emotion_model.predict(validation_generator), axis=1))

# Convert the confusion matrix to numpy array
confusion_mtx_val = confusion_mtx_val.numpy()

# Plot the confusion matrix for the validation set
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_val, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Validation Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save the plot as an image file
plt.savefig(f'results_{keyword}/confusion_matrix_{keyword}_val.png')
plt.close()

# Compute the confusion matrix for the test set
confusion_mtx_test = tf.math.confusion_matrix(test_generator.classes, np.argmax(emotion_model.predict(test_generator), axis=1))

# Convert the confusion matrix to numpy array
confusion_mtx_test = confusion_mtx_test.numpy()

# Plot the confusion matrix for the test set
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_test, annot=True, fmt="d", cmap='Blues')
plt.title(f"Confusion Matrix for {keyword} Test Set")
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save the plot as an image file
plt.savefig(f'results_{keyword}/confusion_matrix_{keyword}_test.png')
plt.close()

# Predict the classes
y_pred = emotion_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the classification report
report = classification_report(test_generator.classes, y_pred_classes, target_names=test_generator.class_indices.keys(), output_dict=True)

# Convert the report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Save the report as a CSV file
report_df.to_csv(f'results_{keyword}/classification_report_{keyword}.csv')


# Get the history of the training process
history = emotion_model_info.history
# Create a DataFrame from the history
df = pd.DataFrame(history)
# Save the DataFrame as a CSV file
df.to_csv(f'results_{keyword}/training_history_{keyword}.csv', index=False)


#Save the model structure in json file
model_json = emotion_model.to_json()
with open(f"results_{keyword}/model_{keyword}.json", "w") as json_file:
    json_file.write(model_json)

#Save the model
emotion_model.save_weights(f'results_{keyword}/emotion_model_{keyword}.h5')