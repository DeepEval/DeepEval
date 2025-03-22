import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    SeparableConv2D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    Dense,
    Add,
)

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Depthwise separable convolutional layer
    conv_out = SeparableConv2D(
        filters=64,
        kernel_size=7,
        padding="same",
        use_bias=False,
        name="conv_out",
    )(inputs)

    # Layer normalization
    norm_out = BatchNormalization(name="norm_out")(conv_out)

    # Activation layer
    act_out = Activation("relu", name="act_out")(norm_out)

    # Global average pooling
    pool_out = GlobalAveragePooling2D()(act_out)

    # Fully connected layer 1
    fc1 = Dense(units=64, name="fc1")(pool_out)

    # Fully connected layer 2
    fc2 = Dense(units=64, name="fc2")(fc1)

    # Add operation
    added = Add(name="added")([inputs, fc2])

    # Output layer
    outputs = Dense(units=10, activation="softmax", name="outputs")(added)

    # Model creation
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Model compilation
model = dl_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])