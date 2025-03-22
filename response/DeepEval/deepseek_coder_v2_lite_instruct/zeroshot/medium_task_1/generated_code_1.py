import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Second convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Add the input to the output of the second convolutional layer
    x = Add()([x, inputs])
    
    # Max-pooling layer
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten the features
    x = Flatten()(x)
    
    # First fully connected layer
    x = Dense(128, activation='relu')(x)
    
    # Second fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()