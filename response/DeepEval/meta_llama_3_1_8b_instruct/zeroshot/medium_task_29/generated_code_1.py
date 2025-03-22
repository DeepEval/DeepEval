# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.

    The model includes three max pooling layers with varying window sizes to extract features.
    The pooling windows are 1x1, 2x2, and 4x4, with the stride set equal to the size of the respective pooling window.
    The output from each pooling layer is flattened into a one-dimensional vector, and these vectors are concatenated to form a unified feature set.
    Finally, the combined features are fed into two fully connected layers to produce classification results for the 10 classes.

    Returns:
        A compiled Keras model.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Convolutional layer with 32 filters and kernel size 3x3
    x = Conv2D(32, (3, 3), activation='relu')(inputs)

    # First max pooling layer with window size 1x1 and stride 1x1
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)

    # Second max pooling layer with window size 2x2 and stride 2x2
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Third max pooling layer with window size 4x4 and stride 4x4
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten the output of each pooling layer
    x1 = Flatten()(x)
    x2 = Flatten()(x)
    x3 = Flatten()(x)

    # Concatenate the flattened vectors to form a unified feature set
    x = Concatenate()([x1, x2, x3])

    # First fully connected layer with 128 units and ReLU activation
    x = Dense(128, activation='relu')(x)

    # Second fully connected layer with 10 units and softmax activation for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the Keras model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a suitable loss function, optimizer, and metrics
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model