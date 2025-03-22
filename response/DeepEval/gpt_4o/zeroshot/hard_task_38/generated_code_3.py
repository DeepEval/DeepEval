import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    def block(x, filters):
        # Batch Normalization
        x_bn = BatchNormalization()(x)
        # ReLU Activation
        x_relu = ReLU()(x_bn)
        # Convolutional Layer
        x_conv = Conv2D(filters, (3, 3), padding='same')(x_relu)
        # Concatenate original input with the new features
        x_out = Concatenate()([x, x_conv])
        return x_out

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    x1 = input_layer
    for _ in range(3):
        x1 = block(x1, filters=32)

    # Pathway 2
    x2 = input_layer
    for _ in range(3):
        x2 = block(x2, filters=32)

    # Concatenate the outputs from both pathways
    x = Concatenate()([x1, x2])

    # Flatten the concatenated outputs
    x = Flatten()(x)

    # Fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()