# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins with an initial convolutional layer that adjusts the number of output channels to match the input image channels.
    Block 1 features two parallel processing paths: Path1: Global average pooling followed by two fully connected layers.
    Path2: Global max pooling followed by two fully connected layers. These paths both extract features whose size is equal to the input's channels.
    The outputs from both paths are added and passed through an activation function to generate channel attention weights matching the input's shape,
    which are then applied to the original features through element-wise multiplication.
    Block 2 extracts spatial features by separately applying average pooling and max pooling. The outputs are concatenated along the channel dimension,
    followed by a 1x1 convolution and a sigmoid activation to normalize the features. These normalized features are then multiplied element-wise with the channel dimension features from Block 1.
    Finally, an additional branch with a 1x1 convolutional layer ensures the output channels align with the input channels. The result is added to the main path and activated.
    The final classification is performed through a fully connected layer.
    
    Parameters:
    None
    
    Returns:
    model (tf.keras.Model): The constructed deep learning model.
    """

    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1: Channel attention mechanism
    path1 = layers.GlobalAveragePooling2D()(x)
    path1 = layers.Dense(128, activation='relu')(path1)
    path1 = layers.Dense(128, activation='relu')(path1)

    path2 = layers.GlobalMaxPooling2D()(x)
    path2 = layers.Dense(128, activation='relu')(path2)
    path2 = layers.Dense(128, activation='relu')(path2)

    # Combine the two paths and apply activation function
    x = layers.Add()([path1, path2])
    x = layers.Activation('sigmoid')(x)

    # Apply channel attention weights to the original features
    x = layers.Lambda(lambda z: z * x)(x)

    # Block 2: Spatial attention mechanism
    avg_pool = layers.AveragePooling2D((2, 2))(x)
    avg_pool = layers.Conv2D(32, (1, 1))(avg_pool)
    avg_pool = layers.BatchNormalization()(avg_pool)
    avg_pool = layers.Activation('relu')(avg_pool)

    max_pool = layers.MaxPooling2D((2, 2))(x)
    max_pool = layers.Conv2D(32, (1, 1))(max_pool)
    max_pool = layers.BatchNormalization()(max_pool)
    max_pool = layers.Activation('relu')(max_pool)

    # Concatenate the spatial features and apply sigmoid activation
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Conv2D(32, (1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)

    # Apply spatial attention weights to the channel dimension features
    x = layers.Lambda(lambda z: z * x)(x)

    # Additional branch to align output channels with input channels
    x = layers.Conv2D(3, (1, 1))(x)

    # Add the result of the additional branch to the main path
    x = layers.Add()([x, input_layer])

    # Activation function
    x = layers.Activation('relu')(x)

    # Final fully connected layer for classification
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model