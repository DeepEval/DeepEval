import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Sequential convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

        # Parallel path
        parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Addition of paths
        add1 = Add()([conv1, parallel_conv])
        add2 = Add()([conv2, add1])
        add3 = Add()([conv3, add2])

        return add3

    # Two parallel branches using the block
    branch1_output = block(input_layer)
    branch2_output = block(input_layer)

    # Concatenate outputs of both branches
    concatenated = Concatenate()([branch1_output, branch2_output])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model