import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

# Define labels and image size
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# Function to load and preprocess the data
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return data

# Load the data
data_dir = 'assests/train'  # Update this path
data = get_training_data(data_dir)

# Split the data into features and labels
X = []
y = []

for feature, label in data:
    X.append(feature)
    y.append(label)

# Convert lists to numpy arrays
X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

# Normalize the pixel values
X = X / 255.0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
# model compiling
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image size and labels
img_size = 150

# Function to load and preprocess test images
def load_and_preprocess_image(image_path):
    try:
        img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshape image to preferred size
        return resized_arr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Directory containing new data images
new_data_dir = 'assests/Predict'  # Update this path
new_images = []
new_image_paths = []

for img in os.listdir(new_data_dir):
    image_path = os.path.join(new_data_dir, img)
    preprocessed_image = load_and_preprocess_image(image_path)
    if preprocessed_image is not None:
        new_images.append(preprocessed_image)
        new_image_paths.append(image_path)

# Convert the list to a NumPy array and normalize pixel values
X_new = np.array(new_images).reshape(-1, img_size, img_size, 1) / 255.0

# Make predictions on the new images
new_predictions = model.predict(X_new)

# Function to map prediction probability to label
def get_label(prediction, threshold=0.5):
    return 'PNEUMONIA' if prediction >= threshold else 'NORMAL'

# Display new images with predicted labels
for i in range(len(X_new)):
    prediction = new_predictions[i]
    
    image_label = get_label(prediction)
    print(f'Image: {os.path.basename(new_image_paths[i])}, Prediction: {image_label}, Probability: {prediction[0]}')

    # Optionally, display the image and prediction using matplotlib
    plt.imshow(X_new[i].reshape(img_size, img_size), cmap='gray')
    plt.title(f'Prediction: {image_label} ({prediction[0]:.2f})')
    plt.show()

import pickle as pkl
with open('model.pickle', 'wb') as file:
    pkl.dump(model, file)

# Save the trained model
model.save('pneumonia_detection_model.h5', save_format='h5')