import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, AveragePooling2D
from tensorflow.keras import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into 3 groups
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        # Process each group with a 1x1 convolutional layer
        processed_groups = [Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in splits]
        # Concatenate the processed groups
        output = Concatenate(axis=-1)(processed_groups)
        return output

    def block2(x):
        # Get the shape of the input
        height, width, channels = x.shape[1:]
        # Reshape the input to target shape
        x = tf.reshape(x, (height, width, 3, int(channels / 3)))
        # Swap the third and fourth dimensions
        x = Permute((1, 2, 4, 3))(x)
        # Reshape back to original shape
        x = tf.reshape(x, (height, width, channels))
        return x

    def block3(x):
        # Apply 3x3 depthwise separable convolution
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', depthwise_mode=True)(x)
        return x

    # Apply Block 1
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)

    # Repeat Block 1
    x = block1(x)
    x = block2(x)
    x = block3(x)

    # Branch path with average pooling
    branch = AveragePooling2D(pool_size=(8, 8), strides=1)(input_layer)
    branch = Flatten()(branch)

    # Concatenate main path and branch path outputs
    combined = Concatenate(axis=-1)([x, branch])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()