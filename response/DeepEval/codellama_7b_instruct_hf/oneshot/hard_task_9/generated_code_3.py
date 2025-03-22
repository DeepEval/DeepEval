from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Define the first branch with a 1x1 convolution
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Define the second branch with a 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)

    # Define the third branch with a 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
    branch3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Apply a 1x1 convolution to adjust the output dimensions
    output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)

    # Add the branch directly connected to the input
    output = Add()([output, input_layer])

    # Define the main path with a MaxPooling2D layer
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)

    # Apply BatchNormalization and Flatten layers
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)

    # Define the classification layers
    dense1 = Dense(units=128, activation='relu')(main_path)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model