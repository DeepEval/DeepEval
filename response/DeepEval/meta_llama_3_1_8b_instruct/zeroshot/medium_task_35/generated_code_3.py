from tensorflow.keras import layers, models
from tensorflow.keras import activations
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def dl_model():
    # Define the input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Convolutional and max pooling stage 1
    x = layers.Conv2D(32, (3, 3), activation=activations.relu, padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional and max pooling stage 2
    x = layers.Conv2D(64, (3, 3), activation=activations.relu, padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional and dropout stage
    x = layers.Conv2D(128, (3, 3), activation=activations.relu, padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Convolutional and dropout stage
    x = layers.Conv2D(128, (3, 3), activation=activations.relu, padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Upsampling stage 1
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, layers.Conv2D(128, (3, 3), activation=activations.relu, padding='same')(inputs)])

    # Convolutional and dropout stage
    x = layers.Conv2D(128, (3, 3), activation=activations.relu, padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Upsampling stage 2
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, layers.Conv2D(64, (3, 3), activation=activations.relu, padding='same')(inputs)])

    # Convolutional and dropout stage
    x = layers.Conv2D(64, (3, 3), activation=activations.relu, padding='same')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    # Define the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model