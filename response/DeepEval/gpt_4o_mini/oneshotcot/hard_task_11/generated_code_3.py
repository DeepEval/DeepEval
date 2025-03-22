import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway with 1x1 convolution
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel path with convolutions
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of the parallel paths
    parallel_output = Concatenate()([path1, path2, path3])

    # Combine main path and parallel output with another 1x1 convolution
    combined_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path + parallel_output)

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model