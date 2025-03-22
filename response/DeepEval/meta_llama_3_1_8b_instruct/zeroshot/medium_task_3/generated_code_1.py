# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function generates a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two sequential blocks, each comprising a convolutional layer followed by a max pooling layer, 
    which extracts deep features from the input. The last convolution output dimension needs to be the same as the channel 
    dimension of the input image. The outputs from both paths are then combined with the input through an addition operation, 
    creating a comprehensive feature representation. Finally, a fully connected layer classifies the output, providing a probability 
    distribution across ten categories.
    """

    # Define the input shape for the MNIST dataset, which consists of 28x28 grayscale images
    input_shape = (28, 28, 1)

    # Create the base model using the Functional API
    inputs = keras.Input(shape=input_shape)

    # Convolutional and Max Pooling block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolutional and Max Pooling block 2
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Add the input layer back to the output of the convolutional and max pooling blocks
    # This creates a comprehensive feature representation
    x = layers.Add()([inputs, x])

    # Flatten the output of the convolutional and max pooling blocks
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model