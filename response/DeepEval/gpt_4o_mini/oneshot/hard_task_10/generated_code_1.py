import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: sequence of convolutions (1x1 -> 1x7 -> 7x1)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenate the outputs of the two paths
    concatenated = keras.layers.Concatenate()([path1, path2])

    # 1x1 convolution to align output dimensions
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Adding the main output to the original input (skip connection)
    merged_output = Add()([main_output, input_layer])

    # Flattening the result for the fully connected layers
    flatten_layer = Flatten()(merged_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model