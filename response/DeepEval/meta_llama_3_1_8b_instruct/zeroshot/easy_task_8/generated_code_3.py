# Import necessary packages
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model begins with a depthwise separable convolutional layer to process the input data,
    followed by a 1x1 convolutional layer for feature extraction. To mitigate overfitting,
    a dropout layer is added after each convolutional layer. Finally, the output is flattened
    and passed through a fully connected layer to generate classification probabilities.
    
    Returns:
        A compiled Keras model.
    """

    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)  # 28x28 grayscale images

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Add the first depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)  # Add dropout to mitigate overfitting

    # Add the 1x1 convolutional layer for feature extraction
    x = Conv2D(kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)  # Add dropout to mitigate overfitting

    # Flatten the output for the fully connected layer
    x = Flatten()(x)

    # Add the fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model