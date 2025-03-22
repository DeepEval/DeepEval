import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path with a single 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path with a sequence of convolutions: 1x1, 1x7, and 7x1
    path2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_2)

    # Concatenate outputs of the two paths
    concatenated = Concatenate()([path1, path2_3])

    # Apply a 1x1 convolution to align dimensions
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path that connects directly to the input
    branch_path = input_layer

    # Merge the outputs of the main path and the branch path through addition
    merged = Add()([main_path, branch_path])

    # Flatten the result and pass through fully connected layers for classification
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model