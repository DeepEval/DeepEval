import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape, Lambda

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Concatenate()([path1, path2, path3])

    # Block 2
    block2_output = Reshape(target_shape=(14, 14, 4, 16))(block1_output)
    block2_output = Permute(dims=(0, 1, 3, 2))(block2_output)
    block2_output = Reshape(target_shape=(14, 14, 16))(block2_output)

    # Final output
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model