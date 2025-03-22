from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with a single channel

    # 1x1 Convolutional layer to increase dimensionality
    x = Conv2D(32, (1, 1), strides=2, activation='relu')(inputs)

    # 3x3 Depthwise Separable Convolutional layer for feature extraction
    x = SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)

    # 1x1 Convolutional layer to reduce dimensionality
    x = Conv2D(32, (1, 1), strides=2, activation='relu')(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)  # MNIST has 10 classes

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()