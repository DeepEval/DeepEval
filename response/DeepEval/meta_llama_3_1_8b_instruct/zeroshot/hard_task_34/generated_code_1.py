# Import necessary packages
from keras.layers import Input, ReLU, Conv2D, Conv2DTranspose, Concatenate, Flatten, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import Activation
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.regularizers import l2
import tensorflow as tf

def dl_model():
    # Define the main pathway
    def main_path(inputs):
        x = inputs
        for _ in range(3):
            # Apply a separable convolutional layer with ReLU activation
            x = SeparableConv2D(32, (3, 3), padding='same')(x)
            x = ReLU()(x)
            x = AveragePooling2D((2, 2))(x)
            x = Concatenate()([x, x])
        return x

    # Define the branch pathway
    def branch_path(inputs):
        x = Conv2D(32, (1, 1), padding='same')(inputs)
        x = ReLU()(x)
        return x

    # Define the main pathway input
    inputs = Input(shape=(28, 28, 1))

    # Pass the input through the main pathway
    main_output = main_path(inputs)

    # Define the branch pathway input
    branch_inputs = main_output

    # Pass the input through the branch pathway
    branch_output = branch_path(branch_inputs)

    # Fuse the two pathways by concatenation along the channel dimension
    x = Concatenate()([main_output, branch_output])

    # Apply a flatten layer to the concatenated features
    x = Flatten()(x)

    # Apply a fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Construct the model
model = dl_model()
print(model.summary())