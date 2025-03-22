import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    main_path_conv1x1_parallel = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_conv1x1)
    main_path_conv3x3_parallel = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1x1)

    # Concatenate the outputs of the parallel convolutions in the main path
    main_path_output = keras.layers.Concatenate()([main_path_conv1x1_parallel, main_path_conv3x3_parallel])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from the main path and the branch path
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model