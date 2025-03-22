import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: A 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: Sequence of convolutions - 1x1, followed by 1x7, and then 7x1
    path2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_1x1)
    path2_7x1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_1x7)

    # Concatenate outputs from the two paths
    concatenated_paths = Concatenate()([path1, path2_7x1])

    # Apply a 1x1 convolution to align the output dimensions
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_paths)

    # Branch that connects directly to the input
    branch_output = input_layer

    # Merge the outputs of the main path and the branch through addition
    merged_output = Add()([main_path_output, branch_output])

    # Flatten the result for the fully connected layers
    flatten_layer = Flatten()(merged_output)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model