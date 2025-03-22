import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape based on the MNIST dataset
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels and have 1 color channel (grayscale)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1x1 Convolution to reduce input dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 Convolution to extract features
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    # 1x1 Convolution to restore dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer with 10 neurons for classification
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()