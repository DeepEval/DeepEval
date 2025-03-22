import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Dense, Flatten, Concatenate
from tensorflow.keras import backend as K

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (1, 1), activation='relu')(x)

    # Branch path
    branch = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Addition of main and branch paths
    x = Add()([x, branch])

    # Second block
    # Split the input into three groups
    split_points = [1, 2]
    x = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=split_points, axis=-1))(x)

    # Extract features using depthwise separable convolutional layers with different kernel sizes
    def depthwise_conv(kernel_size):
        return Conv2D(1, kernel_size, padding='same', depthwise_constraint=tf.keras.constraints.max_norm(1.))(x)

    outputs = []
    for kernel_size in [(1, 1), (3, 3), (5, 5)]:
        output = depthwise_conv(kernel_size)
        output = Conv2D(32, (1, 1), activation='relu')(output)
        outputs.append(output)

    # Concatenate outputs
    x = Concatenate()(outputs)

    # Flatten and add fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create and return the model
    model = Model(inputs=inputs, outputs=x)
    return model

# Example usage:
# model = dl_model()
# model.summary()