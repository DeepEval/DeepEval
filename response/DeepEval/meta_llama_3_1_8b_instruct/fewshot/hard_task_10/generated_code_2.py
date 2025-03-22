import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has images of size 32x32 with 3 color channels

    # Define the main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the branch path with 1x1, 1x7, and 7x1 convolutions
    branch_path_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(branch_path_1)
    branch_path_3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(branch_path_2)

    # Concatenate the outputs of the main path and the branch path
    concatenated_path = Concatenate()([main_path, branch_path_3])

    # Apply a 1x1 convolution to align the output dimensions
    output_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_path)

    # Merge the main path and the branch path through addition
    merged_path = Add()([output_path, input_layer])

    # Flatten the output
    flatten_layer = Flatten()(merged_path)

    # Define the fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model