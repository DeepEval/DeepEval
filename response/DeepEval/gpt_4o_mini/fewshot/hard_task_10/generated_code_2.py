import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Sequence of convolutions (1x1, 1x7, 7x1)
    path2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2_conv1)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_conv2)

    # Concatenate outputs of both paths
    concatenated = Concatenate()([path1, path2])

    # 1x1 convolution to align output dimensions with the input image's channel
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch that connects directly to the input
    branch_output = input_layer

    # Merge outputs of main path and branch through addition
    merged_output = Add()([main_path_output, branch_output])

    # Flatten the output and pass through two fully connected layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model