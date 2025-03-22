import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization

def dl_model():
    inputs = Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Stage 2: More Convolution and Max Pooling
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Additional Convolutional Layers
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)
    
    # Upsampling and Skip Connections
    deconv1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop3)
    deconv1 = concatenate([deconv1, conv2])
    deconv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv1)
    deconv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv1)
    
    deconv2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv1)
    deconv2 = concatenate([deconv2, conv1])
    deconv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv2)
    deconv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv2)
    
    # Output Layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(deconv2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()