import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Activation
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Initial Convolution + Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    # Stage 2: Convolution + Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Additional Convolution + Dropout Layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    dropout1 = Dropout(rate=0.5)(conv3)
    
    # Upsampling Stage 1
    up1 = UpSampling2D(size=(2, 2))(dropout1)
    skip1 = Concatenate()([up1, conv2])
    up_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    
    # Upsampling Stage 2
    up2 = UpSampling2D(size=(2, 2))(up_conv1)
    skip2 = Concatenate()([up2, conv1])
    up_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    
    # Output Layer: 1x1 Convolution for classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), padding='same')(up_conv2)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model