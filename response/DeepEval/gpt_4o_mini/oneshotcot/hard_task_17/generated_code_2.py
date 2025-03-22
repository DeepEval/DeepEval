import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Global average pooling layer
    gap = GlobalAveragePooling2D()(input_layer)
    # Fully connected layer 1
    dense1 = Dense(units=32, activation='relu')(gap)
    # Fully connected layer 2
    dense2 = Dense(units=32, activation='relu')(dense1)
    # Reshape to match input shape (32, 32, 3)
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    # Multiply input with the reshaped weights
    weighted_output = Multiply()([input_layer, reshaped_weights])

    # Block 2
    # Convolutional layer 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(weighted_output)
    # Convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Adding the outputs from Block 1 and Block 2
    # First, we need to ensure the shapes match; we can use a Conv2D layer to modify the shape of the output from Block 1
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(weighted_output)

    # Merge the outputs
    combined_output = Add()([branch_output, max_pooling])

    # Fully connected layers after merging
    flatten = Flatten()(combined_output)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model