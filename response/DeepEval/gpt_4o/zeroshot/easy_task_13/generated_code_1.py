import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with a single channel
    inputs = Input(shape=input_shape)
    
    # First 1x1 Convolutional Layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)
    x = Dropout(0.2)(x)
    
    # Second 1x1 Convolutional Layer
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 3x1 Convolutional Layer
    x = Conv2D(32, (3, 1), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    
    # 1x3 Convolutional Layer
    x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    
    # 1x1 Convolutional Layer to match the input channel count
    x = Conv2D(1, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Add the processed features with the original input
    x = Add()([x, inputs])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer to produce the final output
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of how to create the model
model = dl_model()
model.summary()