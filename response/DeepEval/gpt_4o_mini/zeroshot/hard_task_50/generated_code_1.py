import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 32x32 RGB images (CIFAR-10)
    input_layer = layers.Input(shape=(32, 32, 3))

    # First Block
    # Max Pooling with 1x1
    x1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    x1_flat = layers.Flatten()(x1)

    # Max Pooling with 2x2
    x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    x2_flat = layers.Flatten()(x2)

    # Max Pooling with 4x4
    x3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    x3_flat = layers.Flatten()(x3)

    # Concatenating the flattened vectors
    concatenated = layers.concatenate([x1_flat, x2_flat, x3_flat])

    # Dropout layer to mitigate overfitting
    dropout = layers.Dropout(0.5)(concatenated)

    # Fully connected layer
    dense_layer = layers.Dense(128, activation='relu')(dropout)

    # Reshape into a 4D tensor (BATCH_SIZE, 1, 1, 128)
    reshaped = layers.Reshape((1, 1, 128))(dense_layer)

    # Second Block
    # Split the tensor into 4 groups along the last dimension
    split_groups = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each group with separable convolutions
    conv_outputs = []
    for kernel_size in [1, 3, 5, 7]:
        conv = layers.SeparableConv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_groups[0])
        conv_outputs.append(conv)

    # Concatenate outputs from the separable convolutions
    concatenated_conv = layers.concatenate(conv_outputs)

    # Flatten the final output
    flattened_output = layers.Flatten()(concatenated_conv)

    # Final output layer for classification with softmax activation
    output_layer = layers.Dense(10, activation='softmax')(flattened_output)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of using the model
model = dl_model()
model.summary()