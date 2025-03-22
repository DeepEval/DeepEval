import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Branch 1: Local feature extraction
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)
    
    # Branch 2: Sequential layers for downsampling and feature extraction
    branch2 = Conv2D(64, (3, 3), activation='relu')(MaxPooling2D((2, 2))(x))
    branch2 = UpSampling2D((2, 2))(branch2)
    
    # Branch 3: Sequential layers for downsampling and feature extraction
    branch3 = Conv2D(64, (3, 3), activation='relu')(MaxPooling2D((2, 2))(x))
    branch3 = UpSampling2D((2, 2))(branch3)
    
    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Further processing through a 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(concatenated)
    
    # Flatten the output for fully connected layers
    x = Flatten()(x)
    
    # Three fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()