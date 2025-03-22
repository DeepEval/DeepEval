import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Define the main path
    def main_path(input_tensor):
        relu1 = keras.layers.ReLU()(input_tensor)
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu1)
        max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sep_conv1)

        relu2 = keras.layers.ReLU()(max_pooling1)
        sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu2)
        max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sep_conv2)

        return max_pooling2
    
    main_path_output = main_path(input_layer)

    # Define the branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(input_layer)

    # Sum the outputs from both paths
    summed_output = Concatenate()([main_path_output, branch_path_output])

    # Batch normalization
    bath_norm = BatchNormalization()(summed_output)

    # Flatten the result
    flatten_layer = Flatten()(bath_norm)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model