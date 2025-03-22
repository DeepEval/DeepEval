import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main pathway - single 1x1 convolution
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel branch with 1x1, 1x3, and 3x1 convolutions
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1)
    branch_conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv2)

    # Concatenation of main path and branch path
    concatenated = Concatenate()([main_path_conv1, branch_conv3])

    # Another 1x1 convolution to produce the main output
    final_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Additive fusion of the input and final convolution output
    added = Add()([input_layer, final_conv])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model