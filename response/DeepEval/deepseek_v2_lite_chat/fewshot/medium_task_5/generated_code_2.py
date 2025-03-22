import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate
from keras.layers import BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block_main(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(bn1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(bn2)

        return pool2

    def block_branch(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(bn1)
        return pool1

    block_main_output = block_main(input_layer)
    block_branch_output = block_branch(input_layer)

    add_layer = Add()([block_main_output, block_branch_output])

    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model