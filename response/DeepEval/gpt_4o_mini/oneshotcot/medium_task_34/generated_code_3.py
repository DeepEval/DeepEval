import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First Part: Convolutional and Max-Pooling Layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Second Part: Convolutional Layers with Dropout
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    drop1 = Dropout(rate=0.5)(conv4)
    
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(drop1)
    
    # Third Part: Upsampling with Skip Connections
    upsample1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    skip1 = Concatenate()([upsample1, conv3])
    
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    upsample2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    skip2 = Concatenate()([upsample2, conv2])
    
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    upsample3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    skip3 = Concatenate()([upsample3, conv1])
    
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip3)
    
    # Final 1x1 Convolutional Layer for Output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv8)
    
    # Create Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model