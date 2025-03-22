import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # 1x1 Convolutional Layer to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 Depthwise Separable Convolutional Layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
    
    # Another 1x1 Convolutional Layer to reduce dimensionality
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    
    # Add the processed output to the original input layer
    x = Add()([x, inputs])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer to generate the final classification probabilities
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()