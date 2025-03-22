import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense
from tensorflow.keras.layers import DepthwiseConv2D, ReLU

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)

    # Another 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Element-wise addition of the processed output to the original input
    x = Add()([x, inputs])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer to generate the final classification probabilities
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()