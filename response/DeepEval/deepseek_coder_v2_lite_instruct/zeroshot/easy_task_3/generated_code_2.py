import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape, name="input_layer")
    
    # First block: Convolution, Convolution, Max Pooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Second block: Convolution, Convolution, Convolution, Max Pooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
    
    return model

# Example usage
model = dl_model()
model.summary()