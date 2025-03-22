import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(path_2)
        output_tensor = Concatenate()([path_1, path_2, path_3])
        return output_tensor

    # Transition convolution
    def transition_conv(input_tensor, num_filters):
        x = Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        return x

    # Block 2
    def block_2(input_tensor):
        # Global max pooling
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)

        # Channel-matching weights
        shape_before_pool = keras.backend.int_shape(x)
        x = Flatten()(x)
        x = Dense(units=shape_before_pool[1] * shape_before_pool[2] * shape_before_pool[3])(x)
        x = Lambda(lambda y: K.reshape(y, (-1, shape_before_pool[1], shape_before_pool[2], shape_before_pool[3])))(x)

        # Reshape and multiply with input
        shape_before_pool = keras.backend.int_shape(x)
        y = Lambda(lambda y: K.reshape(y, (-1, shape_before_pool[1] * shape_before_pool[2], shape_before_pool[3])))(x)
        y = Dense(units=shape_before_pool[3])(y)
        y = Lambda(lambda y: K.reshape(y, (-1, shape_before_pool[1], shape_before_pool[2], shape_before_pool[3])))(y)
        y = Conv2D(filters=shape_before_pool[3], kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)

        # Main path output and branch output
        output_tensor = x + y

        # Batch normalization and activation
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = keras.activations.relu(output_tensor)

        return output_tensor

    # Main path
    main_output = block_1(input_layer)
    main_output = transition_conv(main_output, 128)
    main_output = block_2(main_output)

    # Branch output
    branch_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    branch_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_output)

    # Final output
    output_tensor = keras.layers.Add()([main_output, branch_output])
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model