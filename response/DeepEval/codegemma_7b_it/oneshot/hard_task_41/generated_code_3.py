import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    avgpool1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avgpool2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avgpool4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    avgpool1x1_flatten = Flatten()(avgpool1x1)
    avgpool2x2_flatten = Flatten()(avgpool2x2)
    avgpool4x4_flatten = Flatten()(avgpool4x4)
    avgpool_dropout = Dropout(rate=0.5)(Concatenate()([avgpool1x1_flatten, avgpool2x2_flatten, avgpool4x4_flatten]))

    # Block 2
    branch1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1x1_pool = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer))
    branch_concat = Concatenate()([branch1x1, branch3x3, branch5x5, branch1x1_pool])

    # Fusion layers
    fusion_avgpool = Concatenate()([avgpool_dropout, branch_concat])
    fusion_dense = Dense(units=256, activation='relu')(fusion_avgpool)

    # Final layers
    reshape_layer = Reshape((4, 4, 256))(fusion_dense)
    dense1 = Dense(units=256, activation='relu')(reshape_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model