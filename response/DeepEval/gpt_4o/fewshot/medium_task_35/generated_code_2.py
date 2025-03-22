import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Stage 1: Initial Convolution and Pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Stage 2: Additional Convolution and Pooling
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Processing through additional convolution and dropout layers
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    drop1 = Dropout(rate=0.5)(conv3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(drop1)
    drop2 = Dropout(rate=0.5)(conv4)

    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(drop2)
    skip1 = Add()([up1, conv2])
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(skip1)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    skip2 = Add()([up2, conv1])
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip2)

    # Final 1x1 convolution to produce class probabilities
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(conv6)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model