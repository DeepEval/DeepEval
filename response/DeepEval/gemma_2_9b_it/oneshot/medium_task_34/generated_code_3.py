import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, TransposedConv2D, Dropout, concatenate, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.models import Model

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3)
    dropout = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Part 3: Upsampling and Reconstruction
    up1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    merge1 = concatenate([up1, conv2], axis=3)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge1)

    up2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv6)
    merge2 = concatenate([up2, conv1], axis=3)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge2)

    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv7)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model