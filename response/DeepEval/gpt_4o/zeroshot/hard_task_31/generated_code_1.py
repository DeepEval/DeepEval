from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # First Block
    # Main Path
    x_main = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x_main = Dropout(0.3)(x_main)
    x_main = Conv2D(3, (3, 3), padding='same', activation='relu')(x_main)

    # Branch Path
    x_branch = inputs

    # Add paths together
    x = Add()([x_main, x_branch])

    # Second Block
    # Split input into three groups
    def split_tensor(tensor):
        return tf.split(tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_tensor)(x)

    # Group 1: 1x1 convolution
    group1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    group1 = Dropout(0.3)(group1)

    # Group 2: 3x3 convolution
    group2 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    group2 = Dropout(0.3)(group2)

    # Group 3: 5x5 convolution
    group3 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    group3 = Dropout(0.3)(group3)

    # Concatenate outputs of three groups
    x = Concatenate()([group1, group2, group3])

    # Flatten and Dense layers for final output
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()