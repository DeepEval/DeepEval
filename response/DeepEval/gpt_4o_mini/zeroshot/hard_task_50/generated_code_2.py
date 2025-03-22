import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # First block with three max pooling layers with different scales
    max_pool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    max_pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    max_pool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten each max pooling output
    flat1 = layers.Flatten()(max_pool1)
    flat2 = layers.Flatten()(max_pool2)
    flat3 = layers.Flatten()(max_pool3)

    # Apply dropout to mitigate overfitting
    dropout_rate = 0.5
    drop1 = layers.Dropout(dropout_rate)(flat1)
    drop2 = layers.Dropout(dropout_rate)(flat2)
    drop3 = layers.Dropout(dropout_rate)(flat3)

    # Concatenate flattened outputs
    concatenated = layers.concatenate([drop1, drop2, drop3])

    # Fully connected layer and reshape into a 4D tensor
    fc = layers.Dense(256, activation='relu')(concatenated)
    reshaped = layers.Reshape((1, 1, 256))(fc)

    # Second block with separable convolutions
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each split with separable convolutions of different kernel sizes
    conv1 = layers.SeparableConv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = layers.SeparableConv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    conv4 = layers.SeparableConv2D(32, kernel_size=(7, 7), padding='same', activation='relu')(split_inputs[3])

    # Concatenate the outputs from the separable convolutions
    concatenated_conv = layers.concatenate([conv1, conv2, conv3, conv4])

    # Flatten the output and fully connected layer for classification
    flattened_output = layers.Flatten()(concatenated_conv)
    outputs = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()