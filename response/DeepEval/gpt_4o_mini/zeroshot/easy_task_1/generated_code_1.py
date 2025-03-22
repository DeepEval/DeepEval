import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels and grayscale

    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional layer
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    
    # First max pooling layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional layer
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    
    # Flatten the feature maps
    x = layers.Flatten()(x)
    
    # First fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    
    # Second fully connected layer
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer with softmax activation for multi-class classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Optional: If you want to compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()