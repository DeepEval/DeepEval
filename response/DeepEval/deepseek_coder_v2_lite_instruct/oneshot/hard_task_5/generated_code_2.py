import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into 3 groups
        splits = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=-1))(x)
        
        # Process each group with a 1x1 convolutional layer
        processed_splits = [Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in splits]
        
        # Concatenate the processed groups
        x = Concatenate(axis=-1)(processed_splits)
        return x

    # Apply Block 1
    x = block1(input_layer)

    # Block 2: Reshape and Permute
    original_shape = tf.keras.backend.int_shape(x)
    x = Reshape((original_shape[1], original_shape[2], 3, int(original_shape[3] / 3)))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape(original_shape)(x)

    # Block 3: 3x3 Depthwise Separable Convolution
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', depthwise_mode=True)(x)

    # Branch from the input
    branch = input_layer

    # Add the main path and the branch
    x = tf.keras.layers.add([x, branch])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()