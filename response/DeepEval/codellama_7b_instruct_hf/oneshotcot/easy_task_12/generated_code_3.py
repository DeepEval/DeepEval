import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the main path
    main_path = input_layer
    for i in range(2):
        main_path = SeparableConv2D(32, (3, 3), activation='relu')(main_path)
        main_path = MaxPooling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = SeparableConv2D(32, (1, 1), activation='relu')(input_layer)
    branch_path = MaxPooling2D((2, 2))(branch_path)

    # Add the main and branch paths
    merged_path = keras.layers.concatenate([main_path, branch_path], axis=1)

    # Add a flattening layer
    flatten_layer = Flatten()(merged_path)

    # Add a fully connected layer
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model