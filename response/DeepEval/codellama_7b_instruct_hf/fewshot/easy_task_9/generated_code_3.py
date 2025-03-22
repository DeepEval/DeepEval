import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Apply a 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Apply a depthwise separable convolutional layer for feature extraction
    conv2 = DepthwiseSeparableConv2D((3, 3), activation='relu')(conv1)

    # Apply another 1x1 convolutional layer to reduce dimensionality
    conv3 = Conv2D(16, (1, 1), activation='relu')(conv2)

    # Add the processed output to the original input layer
    added_layer = keras.layers.Add()([input_layer, conv3])

    # Flatten the output and pass it through a fully connected layer for classification
    flattened_layer = Flatten()(added_layer)
    output_layer = Dense(10, activation='softmax')(flattened_layer)

    # Create the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model