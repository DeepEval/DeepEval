from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for the CIFAR-10 images
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    x = Conv2D(9, (1, 1), activation='relu')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)

    # Compute channel attention weights
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(9 // 2, activation='relu')(gap)
    fc2 = Dense(9, activation='sigmoid')(fc1)  # Shape matches the channels of initial features

    # Reshape and apply channel attention weights
    attention = Reshape((1, 1, 9))(fc2)
    x = Multiply()([x, attention])

    # Reduce dimensionality with a 1x1 convolution
    x = Conv2D(3, (1, 1), activation='relu')(x)

    # Combine the output with the initial input
    x = Add()([x, inputs])

    # Flatten and create the final fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model