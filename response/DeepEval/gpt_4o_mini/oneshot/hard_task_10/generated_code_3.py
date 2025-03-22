import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization, Activation

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: sequence of convolutions (1x1, 1x7, 7x1)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenating outputs of both paths
    concatenated = keras.layers.Concatenate()([path1, path2])
    
    # Apply a 1x1 convolution to align the output dimensions with the input image's channel
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)

    # Branch to connect directly to the input
    branch_output = input_layer

    # Merging the outputs through addition
    merged_output = Add()([main_path_output, branch_output])
    batch_norm = BatchNormalization()(merged_output)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model