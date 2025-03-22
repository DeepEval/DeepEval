# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features two parallel convolutional branches with convolutional kernels of varying sizes (3x3 and 5x5 respectively)
    to extract features, which are combined through addition. Following this, a global average pooling layer compresses the features,
    leading into two fully connected layers that apply a softmax function to generate attention weights. The output of each branch
    is multiplied by its corresponding weight and added to produce the final weighted output. Lastly, a fully connected layer
    delivers the probability distribution across the 10 classes.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the model
    inputs = keras.Input(shape=input_shape)

    # Convolutional branch with 3x3 kernel
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(128, activation='relu')(x1)

    # Convolutional branch with 5x5 kernel
    x2 = layers.Conv2D(32, (5, 5), activation='relu')(inputs)
    x2 = layers.Conv2D(64, (5, 5), activation='relu')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(128, (5, 5), activation='relu')(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(128, activation='relu')(x2)

    # Combine the outputs of the two branches through addition
    combined = layers.Add()([x1, x2])

    # Apply global average pooling to compress the features
    global_avg_pooled = layers.GlobalAveragePooling1D()(combined)

    # Apply two fully connected layers with softmax activation to generate attention weights
    attention_weights = layers.Dense(128, activation='relu')(global_avg_pooled)
    attention_weights = layers.Dense(10, activation='softmax')(attention_weights)

    # Multiply the output of each branch by its corresponding weight and add to produce the final weighted output
    weighted_output = layers.Lambda(lambda x: x[0] * x[1] + x[2])([x1, attention_weights, x2])

    # Output layer to deliver the probability distribution across the 10 classes
    outputs = layers.Dense(10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model