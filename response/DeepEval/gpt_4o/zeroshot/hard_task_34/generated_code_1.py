from tensorflow.keras.layers import Input, SeparableConv2D, Conv2D, ReLU, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    num_classes = 10           # There are 10 classes in MNIST (digits 0-9)

    # Input layer
    inputs = Input(shape=input_shape)

    # Main path
    x = inputs
    for _ in range(3):
        # ReLU activation
        relu = ReLU()(x)
        # Separable Conv2D layer
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(relu)
        # Concatenate the input with the convolved features
        x = Concatenate(axis=-1)([x, sep_conv])

    # Branch path
    branch = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), padding='same')(inputs)

    # Fuse main path and branch path
    fused = Add()([x, branch])

    # Flatten the features
    flat = Flatten()(fused)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(flat)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model