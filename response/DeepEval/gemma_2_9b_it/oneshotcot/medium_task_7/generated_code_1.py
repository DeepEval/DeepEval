import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: 
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_bn)
    conv2_bn = BatchNormalization()(conv2)

    # Path 2:
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_bn = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_bn)
    conv4_bn = BatchNormalization()(conv4)

    # Path 3:
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5_bn = BatchNormalization()(conv5)

    # Concatenate all paths
    concat_layer = Concatenate()([conv2_bn, conv4_bn, conv5_bn])
    
    # Flatten and dense layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model