import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # 7x7 depthwise convolution
    x = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    # Layer normalization
    x = LayerNormalization()(x)
    # First 1x1 pointwise convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)
    # Second 1x1 pointwise convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)
    
    # Branch path (directly connects to input)
    branch = input_layer
    
    # Combine paths through addition
    combined = Add()([x, branch])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    output = Dense(units=10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Example usage
model = dl_model()
model.summary()