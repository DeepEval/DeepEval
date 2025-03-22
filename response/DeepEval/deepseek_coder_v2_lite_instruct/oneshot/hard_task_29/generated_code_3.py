import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, BatchNormalization, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    def first_block(input_tensor):
        # Main Path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

        # Branch Path
        branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Addition of Main Path and Branch Path
        added = Add()([conv2, branch])
        return added

    block1_output = first_block(input_layer)

    # Second Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    # Flattening and Concatenation
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model