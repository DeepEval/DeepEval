import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D, Reshape, Permute, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into 3 groups
        splits = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))(x)
        # Process each group with a 1x1 convolutional layer
        processed_splits = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        # Concatenate the processed groups
        output = Concatenate(axis=-1)(processed_splits)
        return output

    def block2(x):
        # Get the shape of the input
        input_shape = tf.keras.backend.int_shape(x)
        # Reshape the input to target shape
        x = Reshape((input_shape[1], input_shape[2], 3, int(input_shape[3]/3)))(x)
        # Permute the dimensions
        x = Permute((1, 2, 4, 3))(x)
        # Reshape back to original shape
        x = Reshape((input_shape[1], input_shape[2], int(input_shape[3])))(x)
        return x

    def block3(x):
        return DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)

    # Apply Block 1
    x = block1(input_layer)
    # Apply Block 2
    x = block2(x)
    # Apply Block 3
    x = block3(x)
    # Repeat Block 1
    x = block1(x)
    # Repeat Block 2
    x = block2(x)
    # Repeat Block 3
    x = block3(x)

    # Branch path with Average Pooling
    branch = AveragePooling2D(pool_size=(4, 4))(input_layer)
    branch = Flatten()(branch)
    branch = Dense(128, activation='relu')(branch)

    # Concatenate main path and branch path outputs
    combined = Concatenate(axis=-1)([x, branch])

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()