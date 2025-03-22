import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, TransposedConv2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    # Branch 1: Local Feature Extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2: Average Pooling and Downsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: Transposed Convolution and Upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = TransposedConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch3)

    # Concatenate Branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Refinement and Output Layer
    concat = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    flatten_layer = Flatten()(concat)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model