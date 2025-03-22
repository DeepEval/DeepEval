import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    # Main path
    x_main = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x_main = Dropout(0.2)(x_main)
    x_main = Conv2D(3, (3, 3), padding='same', activation='relu')(x_main)

    # Branch path
    x_branch = inputs

    # Adding the main path and branch path
    x = Add()([x_main, x_branch])

    # Block 2
    # Split the input into 3 groups
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)

    # Group 1: 1x1 separable convolution
    x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_inputs[0])
    x1 = Dropout(0.2)(x1)

    # Group 2: 3x3 separable convolution
    x2 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])
    x2 = Dropout(0.2)(x2)

    # Group 3: 5x5 separable convolution
    x3 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_inputs[2])
    x3 = Dropout(0.2)(x3)

    # Concatenate the outputs of the three groups
    x = Concatenate()([x1, x2, x3])

    # Final layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()  # To print the model architecture