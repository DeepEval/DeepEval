import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Conv2DTranspose, concatenate, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Stage 2: Additional Convolution and Dropout
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    drop4 = Dropout(0.5)(conv4)

    # Stage 3: Upsampling with Skip Connections
    conv5 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(drop4)
    concat5 = concatenate([conv5, conv2])
    conv6 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(concat5)
    concat6 = concatenate([conv6, conv1])

    # Final Convolution and Output Layer
    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(concat6)

    # Create the model
    model = Model(inputs=input_layer, outputs=conv7)

    return model