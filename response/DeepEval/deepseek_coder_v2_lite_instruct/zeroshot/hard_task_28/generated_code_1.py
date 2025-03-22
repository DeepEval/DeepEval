import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, DepthwiseConv2D, LayerNormalization
from tensorflow.keras.models import Model

def dl_model():
    # Define the input tensor
    input_tensor = Input(shape=(32, 32, 3))

    # Main path
    x = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(input_tensor)
    x = LayerNormalization()(x)
    x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1))(x)
    x = LayerNormalization()(x)
    x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1))(x)
    x = LayerNormalization()(x)

    # Branch path
    y = input_tensor

    # Combine outputs of both paths
    z = Add()([x, y])

    # Flatten the output
    z = Flatten()(z)

    # Fully connected layers
    z = Dense(128, activation='relu')(z)
    z = Dense(10, activation='softmax')(z)  # 10 classes for CIFAR-10

    # Define the model
    model = Model(inputs=input_tensor, outputs=z)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()