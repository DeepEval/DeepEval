from keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Concatenate, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 and grayscale
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # Main path
    x = inputs
    for _ in range(3):
        # Separable Convolutional Block
        conv_feature = SeparableConv2D(32, (3, 3), padding='same')(x)
        relu_feature = ReLU()(conv_feature)
        # Concatenate the input with the convolved feature map
        x = Concatenate(axis=-1)([x, relu_feature])

    # Branch path
    branch_output = Conv2D(32, (1, 1), padding='same')(inputs)

    # Fuse features by adding outputs from main and branch paths
    fused_output = Add()([x, branch_output])

    # Flatten and Fully Connected Layer
    flattened = Flatten()(fused_output)
    outputs = Dense(num_classes, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()