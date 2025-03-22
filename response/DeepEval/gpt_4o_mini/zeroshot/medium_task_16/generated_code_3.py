import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input tensor into three groups along the channel dimension
    def split_and_conv(x):
        # Split the input into 3 parts along the channel dimension
        channels = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Apply 1x1 convolution to each group independently
        convolved = [layers.Conv2D(1, (1, 1), activation='relu')(c) for c in channels]
        
        return convolved

    # Apply the custom split and convolution function
    split_groups = layers.Lambda(split_and_conv)(input_layer)

    # Apply average pooling to each group
    pooled_groups = [layers.AveragePooling2D(pool_size=(2, 2))(g) for g in split_groups]

    # Concatenate the pooled groups along the channel dimension
    concatenated = layers.Concatenate(axis=-1)(pooled_groups)

    # Flatten the concatenated feature maps
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    return model

# Example of using the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Display the model architecture