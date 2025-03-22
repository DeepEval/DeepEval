import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution and MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Stage 2: Convolution and MaxPooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Additional Convolution and Dropout Layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    drop1 = Dropout(0.5)(conv3)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.5)(conv4)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(drop2)
    skip1 = Concatenate()([up1, conv2])  # Skip connection from conv2
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip1)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    skip2 = Concatenate()([up2, conv1])  # Skip connection from conv1
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip2)

    # Final output layer with 1x1 convolution
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv6)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model