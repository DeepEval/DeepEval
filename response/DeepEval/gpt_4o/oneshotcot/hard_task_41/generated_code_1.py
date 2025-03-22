import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Conv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)

        path1_drop = Dropout(rate=0.5)(path1_flat)
        path2_drop = Dropout(rate=0.5)(path2_flat)
        path3_drop = Dropout(rate=0.5)(path3_flat)

        block1_output = Concatenate()([path1_drop, path2_drop, path3_drop])
        return block1_output

    block1_output = block1(input_layer)

    # Between Block 1 and Block 2
    fc1 = Dense(units=512, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 32))(fc1)  # Reshape to a suitable 4D shape for block 2

    # Block 2
    def block2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        block2_output = Concatenate()([branch1, branch2, branch3, branch4])
        return block2_output

    block2_output = block2(reshape_layer)

    # Final classification layers
    flatten_output = Flatten()(block2_output)
    fc2 = Dense(units=256, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model