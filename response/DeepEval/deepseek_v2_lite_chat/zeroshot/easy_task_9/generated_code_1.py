import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Input
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Increase dimensionality
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(3, activation='relu')(x)
    
    # Reduce dimensionality
    x = Conv2D(32, (1, 1), activation='relu')(x)
    
    # Add the processed output to the original input
    x = Add()([x, inputs])
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])