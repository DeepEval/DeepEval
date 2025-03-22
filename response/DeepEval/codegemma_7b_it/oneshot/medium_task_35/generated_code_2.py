import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Downsampling Stage 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Downsampling Stage 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Feature Processing
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    drop4 = Dropout(rate=0.2)(conv4)
    
    # Upsampling Stage
    up_sampling1 = UpSampling2D(size=(2, 2))(drop4)
    up_sampling1 = Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up_sampling1)
    skip_connection1 = Concatenate()([up_sampling1, conv2])
    
    up_sampling2 = UpSampling2D(size=(2, 2))(skip_connection1)
    up_sampling2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(up_sampling2)
    skip_connection2 = Concatenate()([up_sampling2, conv1])
    
    # Output Layer
    conv5 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(skip_connection2)
    flatten_layer = Flatten()(conv5)
    
    model = keras.Model(inputs=input_layer, outputs=flatten_layer)

    return model