import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Feature Extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Feature Enhancement
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool2)
    dropout = Dropout(rate=0.25)(conv3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout)

    # Feature Upsampling
    up_conv5 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv4)
    concat5 = Concatenate()([up_conv5, conv2])
    up_conv6 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(concat5)
    concat6 = Concatenate()([up_conv6, conv1])

    # Output Generation
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(concat6)
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax')(conv7)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model