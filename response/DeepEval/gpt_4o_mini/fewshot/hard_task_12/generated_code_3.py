import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    # Step 1: 1x1 convolution for dimensionality reduction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Two parallel convolutional layers
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv3x3_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)

    # Step 3: Concatenate the outputs from the parallel layers
    main_path_output = Concatenate()([conv1x1_main, conv3x3_main])

    # Branch path
    # Step 4: 3x3 convolution to match dimensions
    branch_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 5: Combine main path and branch path using addition
    combined_output = Add()([main_path_output, branch_path_output])

    # Final classification layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model