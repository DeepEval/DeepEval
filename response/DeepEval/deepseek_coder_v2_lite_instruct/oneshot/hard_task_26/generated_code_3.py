import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch 2: Max pooling, 3x3 convolutional layer, and upsampling
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(branch2)

    # Branch 3: Max pooling, 3x3 convolutional layer, and upsampling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(branch3)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer for the main path
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    branch_input = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the main path output and branch path input
    added = Add()([main_output, branch_input])

    # Flatten the output
    flattened = Flatten()(added)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model