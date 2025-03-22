import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Initial 1x1 convolution to process the input features
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    # Branch 2: Max pooling, 3x3 convolution, then upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max pooling, 3x3 convolution, then upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate outputs of all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution after concatenation
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)

    # Branch path with a 1x1 convolution
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the main path and branch path
    combined_path = Add()([main_path_output, branch_path])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model