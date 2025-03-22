import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool

    def block_2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        reshaped = Reshape(target_shape=(1, 1, 64))(dense2)
        return reshaped

    # First block
    first_block_output = block_1(input_layer)

    # Add the original input to the output of the first block
    added_output = Add()([input_layer, first_block_output])

    # Second block
    second_block_output = block_2(added_output)

    # Multiply the refined weights with the original input
    multiplied_output = Multiply()([second_block_output, added_output])

    # Flatten the output and pass it through a fully connected layer for classification
    flatten_layer = Flatten()(multiplied_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model