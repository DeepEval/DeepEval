import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First block: three parallel paths with average pooling of different scales
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    block1_output = block1(input_layer)
    flatten_layer = Flatten()(block1_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    reshape_layer = Reshape((128,))(dense1)  # Reshape to 4-dimensional tensor for block2 processing
    block2_input = reshape_layer

    # Second block: three branches for feature extraction
    def block2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(branch3)
        branch4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])

        return output_tensor

    block2_output = block2(Reshape((1, 1, 128)))(block2_input)
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer2 = Flatten()(bath_norm)
    dense3 = Dense(units=128, activation='relu')(flatten_layer2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model