import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Define first path (1x1 convolution)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Define second path (1x1 -> 1x7 -> 7x1 convolutions)
    path2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_conv1)
    path2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_conv2)

    # Step 4: Concatenate the outputs of both paths
    concatenated = keras.layers.Concatenate()([path1, path2_conv3])

    # Step 5: Apply a 1x1 convolution to align the output dimensions
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 6: Merge the output with the input through addition
    merged_output = Add()([input_layer, main_path_output])

    # Step 7: Flatten the result and apply dense layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Step 8: Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model