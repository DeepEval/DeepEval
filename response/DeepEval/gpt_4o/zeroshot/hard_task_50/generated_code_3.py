import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = Input(shape=(32, 32, 3))

    # First Block: Max Pooling with different scales
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(inputs)

    # Flatten each pooled output
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)

    # Apply Dropout
    drop_1x1 = Dropout(0.5)(flat_1x1)
    drop_2x2 = Dropout(0.5)(flat_2x2)
    drop_4x4 = Dropout(0.5)(flat_4x4)

    # Concatenate pooled outputs
    concatenated = Concatenate()([drop_1x1, drop_2x2, drop_4x4])

    # Fully connected layer and reshape to four-dimensional tensor
    fc = Dense(512, activation='relu')(concatenated)
    reshaped = Reshape((8, 8, 8))(fc)  # Reshape as needed

    # Second Block: Split and process each group with separable convolutions
    # Split the input into four groups along the last dimension
    def split_tensor(tensor):
        return tf.split(tensor, num_or_size_splits=4, axis=-1)

    split_groups = Lambda(split_tensor)(reshaped)

    # Process each group with different kernel sizes
    sep_conv_1x1 = SeparableConv2D(16, (1, 1), activation='relu', padding='same')(split_groups[0])
    sep_conv_3x3 = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(split_groups[1])
    sep_conv_5x5 = SeparableConv2D(16, (5, 5), activation='relu', padding='same')(split_groups[2])
    sep_conv_7x7 = SeparableConv2D(16, (7, 7), activation='relu', padding='same')(split_groups[3])

    # Concatenate outputs from separable convolutions
    concatenated_sep = Concatenate()([sep_conv_1x1, sep_conv_3x3, sep_conv_5x5, sep_conv_7x7])

    # Flatten and fully connected layer for classification
    flattened = Flatten()(concatenated_sep)
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()