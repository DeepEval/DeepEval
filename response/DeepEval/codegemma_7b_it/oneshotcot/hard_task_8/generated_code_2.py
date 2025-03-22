import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_path)

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    concat_path = Concatenate()([primary_path, branch_path])

    # Block 2
    x = Reshape((-1, concat_path.shape[3]))(concat_path)
    x = Reshape((concat_path.shape[1], concat_path.shape[2], -1))(x)
    x = Permute((2, 1, 3))(x)
    x = Reshape((concat_path.shape[2], concat_path.shape[1], -1))(x)

    flatten_layer = Flatten()(x)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model