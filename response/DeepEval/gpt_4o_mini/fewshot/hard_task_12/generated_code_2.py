import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel Convolutional Layers
    main_path_conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
    main_path_conv2_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
    
    # Concatenating the outputs of parallel layers
    main_path_output = Concatenate()([main_path_conv2_1, main_path_conv2_2])

    # Branch path
    branch_path_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Matching dimensions before addition
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same')(branch_path_conv)

    # Combining paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flattening the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model