import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First feature extraction path: 1x1 convolution
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second feature extraction path: sequence of convolutions: 1x1, followed by 1x7, and then 7x1
    conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    # Concatenate outputs from both paths
    concatenated = Concatenate()([conv1_1, conv2_3])

    # Additional branch connecting directly to the input
    branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    adding_layer = Add()([concatenated, branch])

    # Flatten the output for the final fully connected layers
    flattened = Flatten()(adding_layer)

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model