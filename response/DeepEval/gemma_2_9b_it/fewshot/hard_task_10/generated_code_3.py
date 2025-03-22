import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Path 2: 1x7, 7x1 convolutions
    conv1x7_1 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(input_layer)
    conv7x1_1 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv1x7_1)

    # Concatenate outputs of both paths
    merged_features = Concatenate()([conv1x1_1, conv7x1_1])

    # 1x1 convolution to align dimensions
    output_main_path = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(merged_features)

    # Branch connection
    branch_output = input_layer

    # Add outputs of main path and branch
    final_output = Add()([output_main_path, branch_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model