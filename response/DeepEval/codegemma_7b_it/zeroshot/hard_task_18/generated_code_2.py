import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    AveragePooling2D,
    Concatenate,
)

def residual_conv_block(input_tensor, filters, kernel_size=3, stride=1, stage=0, block="a", dim_ordering="tf"):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "_2a",
    )(input_tensor)
    x = BatchNormalization(name=bn_name_base + "_2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "_2b",
    )(x)
    x = BatchNormalization(name=bn_name_base + "_2b")(x)

    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="valid",
            kernel_initializer="he_normal",
            name=conv_name_base + "_1",
        )(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + "_1")(shortcut)
    else:
        shortcut = input_tensor

    x = keras.layers.add([x, shortcut])
    x = Activation("relu")(x)
    return x
    

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # First Block
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
    x = BatchNormalization(name="bn_conv1")(x)
    x = Activation("relu")(x)

    # Second Block
    x = residual_conv_block(input_tensor=x, filters=64, stage=2, block="a")
    x = residual_conv_block(input_tensor=x, filters=64, stage=2, block="b")

    # Third Block
    x = residual_conv_block(input_tensor=x, filters=128, stage=3, block="a", stride=2)
    x = residual_conv_block(input_tensor=x, filters=128, stage=3, block="b")

    # Fourth Block
    x = residual_conv_block(input_tensor=x, filters=256, stage=4, block="a", stride=2)
    x = residual_conv_block(input_tensor=x, filters=256, stage=4, block="b")

    # Fifth Block
    x = residual_conv_block(input_tensor=x, filters=512, stage=5, block="a", stride=2)
    x = residual_conv_block(input_tensor=x, filters=512, stage=5, block="b")

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer for Classification
    predictions = Dense(10, activation="softmax", name="dense_output")(x)

    # Model Definition
    model = Model(inputs=inputs, outputs=predictions)
    return model

# Construct the model
model = resnet32_model()

# Print the model summary
model.summary()