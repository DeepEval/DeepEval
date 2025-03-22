import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Lambda, Concatenate
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Three Average Pooling Layers with different scales
    pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten each pooling result
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)

    # Concatenate the flattened vectors
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layer followed by a reshape operation
    fc = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc)  # Adjust dimensions accordingly

    # Second Block: Split and Depthwise Separable Convolutions
    def split_and_convolve(x):
        # Split into four groups along the last dimension
        splits = tf.split(x, num_or_size_splits=4, axis=-1)

        # Depthwise separable convolutions with different kernel sizes
        conv_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        conv_7x7 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(splits[3])

        return Concatenate()([conv_1x1, conv_3x3, conv_5x5, conv_7x7])

    # Use Lambda layer to encapsulate the custom function
    processed = Lambda(split_and_convolve)(reshaped)

    # Flatten and fully connected layer for classification
    global_pool = GlobalAveragePooling2D()(processed)
    output = Dense(10, activation='softmax')(global_pool)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example usage:
# model = dl_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])