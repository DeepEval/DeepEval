from keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Concatenate, Add, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import BatchNormalization

def create_feature_block(input_tensor, filters, kernel_size=(3, 3)):
    # Applying Separable Convolution and ReLU activation
    x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Concatenating input and feature map along the channel dimension
    x = Concatenate(axis=-1)([input_tensor, x])
    return x

def dl_model():
    # Input layer for MNIST images
    input_tensor = Input(shape=(28, 28, 1))

    # Main path
    x = input_tensor
    for _ in range(3):  # Repeat the feature extraction block three times
        x = create_feature_block(x, filters=32)

    # Branch path with a convolutional layer
    branch_output = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), padding='same')(input_tensor)

    # Fuse main path and branch path through addition
    fused = Add()([x, branch_output])

    # Flatten and Fully Connected Layer for classification
    flat = Flatten()(fused)
    output_tensor = Dense(10, activation='softmax')(flat)

    # Constructing the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

# Example of how to use the model:
model = dl_model()
model.summary()