import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the CIFAR-10 images (32x32 RGB)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    channels = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define a function to create a separable convolutional block
    def separable_conv_block(input_tensor, filter_size):
        x = layers.SeparableConv2D(32, kernel_size=filter_size, padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    # Process each channel group with different filter sizes
    processed_channels = []
    for filter_size in [(1, 1), (3, 3), (5, 5)]:
        processed_channel = separable_conv_block(channels.pop(0), filter_size)
        processed_channels.append(processed_channel)

    # Concatenate the outputs from the three channel groups
    concatenated = layers.Concatenate()(processed_channels)

    # Fully connected layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the function to create the model
model = dl_model()
model.summary()  # This will print the model summary