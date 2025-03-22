import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the first branch
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(branch1)
    branch1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch1)

    # Define the second branch
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
    branch2 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch2)

    # Define the third branch
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    branch3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
    branch3 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch3)

    # Flatten and concatenate the outputs from the three branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Pass through batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)

    # Add fully connected layers for the final classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=output)

    return model

# Instantiate the model
model = dl_model()
model.summary()