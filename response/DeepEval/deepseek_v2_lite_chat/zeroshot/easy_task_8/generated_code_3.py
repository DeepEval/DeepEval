import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same')(x)
    
    # 1x1 convolutional layer for feature extraction
    x = Conv2D(64, (1, 1), activation='relu')(x)
    
    # Dropout layer after the 1x1 convolutional layer
    x = Dropout(0.5)(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()