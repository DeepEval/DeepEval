# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by splitting the input into three groups along the channel dimension by encapsulating tf.split within Lambda layer 
    and applying 1x1 convolutions to each group independently. The number of convolutional kernels for each group is set to one-third of the input channels.
    
    After this, each group undergoes downsampling via an average pooling layer with consistent parameters.
    
    The three resulting groups of feature maps are then concatenated along the channel dimension.
    
    Finally, the concatenated feature maps are flattened into a one-dimensional vector and passed through two fully connected layers for classification.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply 1x1 convolution to each group independently
    group1 = layers.Conv2D(int(32 / 3), (1, 1), activation='relu')(split_inputs[0])
    group2 = layers.Conv2D(int(32 / 3), (1, 1), activation='relu')(split_inputs[1])
    group3 = layers.Conv2D(int(32 / 3), (1, 1), activation='relu')(split_inputs[2])

    # Downsample each group via average pooling layer
    downsampled_group1 = layers.GlobalAveragePooling2D()(group1)
    downsampled_group2 = layers.GlobalAveragePooling2D()(group2)
    downsampled_group3 = layers.GlobalAveragePooling2D()(group3)

    # Concatenate the resulting groups along the channel dimension
    concatenated = layers.Concatenate()([downsampled_group1, downsampled_group2, downsampled_group3])

    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = layers.Flatten()(concatenated)

    # Pass the flattened feature maps through two fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Test the model
model = dl_model()
print(model.summary())