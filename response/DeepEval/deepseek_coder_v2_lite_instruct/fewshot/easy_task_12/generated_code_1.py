import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool2

    main_path_output = main_path(input_layer)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Sum the outputs from both paths
    added_output = Add()([main_path_output, branch_path])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model