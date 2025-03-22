# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    Creates a deep learning model for image classification using Functional APIs of Keras.
    
    The model begins by compressing the input channels with a 1x1 convolutional layer.
    It then expands the features through two parallel convolutional layers, applying 1x1 and 3x3 convolutions, 
    and concatenates the results. Finally, the output feature map is flattened into a one-dimensional vector 
    and passed through two fully connected layers to produce the classification results.
    
    Returns:
        model (Model): The constructed model.
    """

    # Define input layer
    inputs = Input(shape=(32, 32, 64))

    # Compress input channels with a 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Expand features through two parallel convolutional layers
    x_1x1 = Conv2D(32, (1, 1), activation='relu')(x)
    x_3x3 = Conv2D(32, (3, 3), activation='relu')(x)
    x_concat = concatenate([x_1x1, x_3x3])

    # Flatten output feature map into a one-dimensional vector
    x_flat = Flatten()(x_concat)

    # Pass flattened vector through two fully connected layers
    x = Dense(128, activation='relu')(x_flat)
    outputs = Dense(10, activation='softmax')(x)

    # Define and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model