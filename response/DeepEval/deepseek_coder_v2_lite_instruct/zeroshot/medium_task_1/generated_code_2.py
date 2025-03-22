import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Second convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Max-pooling layer
    x = MaxPooling2D((2, 2))(x)
    
    # Add the input layer to the output of the max-pooling layer
    x = Add()([x, input_layer])
    
    # Flatten the features
    x = Flatten()(x)
    
    # First fully connected layer
    x = Dense(128, activation='relu')(x)
    
    # Second fully connected layer
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()