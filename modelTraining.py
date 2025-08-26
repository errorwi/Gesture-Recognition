import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_size=(64,64)
batch_size=32

training_data = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\anush\OneDrive\Documents\GestureRecognition\archive\leapGestRecog\Training Data",
    image_size=img_size,
    batch_size=batch_size,
    shuffle = True,
    validation_split = 0.2,
    subset="training",
    seed=42
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\anush\OneDrive\Documents\GestureRecognition\archive\leapGestRecog\Validation Data",
    image_size=img_size,
    batch_size=batch_size,
    shuffle = True,
    validation_split = 0.2,
    subset = "validation",
    seed = 42
)

class_names = training_data.class_names
print(class_names)

normalize = layers.Rescaling(1./255)
training_data = training_data.map(lambda x,y: (normalize(x),y))
validation_data = validation_data.map(lambda x,y: (normalize(x),y))

model = keras.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128,(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_data,
                    validation_data=validation_data,
                    epochs=10)

