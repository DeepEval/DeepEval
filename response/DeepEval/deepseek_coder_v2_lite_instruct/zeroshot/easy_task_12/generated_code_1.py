import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Main path
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Branch path
    branch = Conv2D(32, (1, 1), padding='same')(inputs)
    
    # Add the main path and branch path
    x = Add()([x, branch])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()