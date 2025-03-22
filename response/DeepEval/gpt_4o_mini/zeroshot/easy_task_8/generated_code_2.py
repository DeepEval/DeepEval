import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape based on MNIST dataset (28x28 grayscale images)
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel (grayscale)
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Depthwise separable convolutional layer
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # Normalization layer
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)  # Dropout to mitigate overfitting
    
    # 1x1 Convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)  # Normalization layer
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)  # Dropout to mitigate overfitting
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout to mitigate overfitting
    
    # Output layer for classification (10 classes for digits 0-9)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0  # Normalize and add channel dimension
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0  # Normalize and add channel dimension
y_train = to_categorical(y_train, num_classes=10)  # One-hot encoding
y_test = to_categorical(y_test, num_classes=10)  # One-hot encoding

# Build the model
model = dl_model()

# Optionally train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)