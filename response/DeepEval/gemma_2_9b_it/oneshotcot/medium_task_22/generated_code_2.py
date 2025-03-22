import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3)) 

    # Branch 1: 3x3 convolutions
    conv_branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch1)

    # Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv
    conv_branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch2)
    conv_branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch2)

    # Branch 3: Max pooling
    pool_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_branch)

    # Concatenate branch outputs
    concat_output = Concatenate()([conv_branch1, conv_branch2, pool_branch])

    # Flatten and dense layers
    flattened = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model