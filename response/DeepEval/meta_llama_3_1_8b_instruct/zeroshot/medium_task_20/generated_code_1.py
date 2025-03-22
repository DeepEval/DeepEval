# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of four parallel convolutional paths: 
    the first path utilizes a 1x1 convolution, 
    the second path features two 3x3 convolutions stacked after a 1x1 convolution, 
    the third path includes a single 3x3 convolution following a 1x1 convolution, 
    and the fourth path starts with max pooling followed by a 1x1 convolution.
    
    The outputs from these paths are concatenated, flattened, and passed through a dense layer with 128 units 
    before reaching the final output layer, which uses softmax activation to classify the input into one of 10 categories.
    """

    # Define the input shape of the images (32x32 pixels with 3 color channels)
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first convolutional path (1x1 convolution)
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path1 = layers.MaxPooling2D((2, 2))(path1)

    # Define the second convolutional path (1x1 convolution -> 3x3 convolution -> 3x3 convolution)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = layers.Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = layers.Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = layers.MaxPooling2D((2, 2))(path2)

    # Define the third convolutional path (1x1 convolution -> 3x3 convolution)
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = layers.Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = layers.MaxPooling2D((2, 2))(path3)

    # Define the fourth convolutional path (max pooling -> 1x1 convolution)
    path4 = layers.MaxPooling2D((2, 2))(inputs)
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of all paths
    merged = layers.Concatenate()([path1, path2, path3, path4])

    # Apply convolutional layers and max pooling to the concatenated output
    merged = layers.Conv2D(64, (3, 3), activation='relu')(merged)
    merged = layers.MaxPooling2D((2, 2))(merged)

    # Flatten the output
    merged = layers.Flatten()(merged)

    # Apply dense layers
    merged = layers.Dense(128, activation='relu')(merged)
    outputs = layers.Dense(10, activation='softmax')(merged)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Call the function to get the model
model = dl_model()
model.summary()