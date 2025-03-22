# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.

    The model comprises two sequential blocks. The first block extracts features through two 3x3 convolutional layers,
    followed by an average pooling layer for smoothing. The input to the first block is combined with the output of the main path via addition.

    In the second block, the main path compresses the feature map using global average pooling to generate channel weights,
    which are then refined through two fully connected layers with the same number of channels as the output of first block.
    After reshaping, these weights are multiplied by the input. Finally, the flattened output is passed through another fully connected layer for classification.
    """

    # Define the input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Create a input layer
    input_layer = Input(shape=input_shape)

    # First block: feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2), strides=2, padding='same')(x)

    # Main path: feature compression and weight refinement
    main_path = GlobalAveragePooling2D()(x)
    main_path = Dense(32, activation='relu')(main_path)
    main_path = Dense(32, activation='relu')(main_path)
    main_path = Reshape((32, 1))(main_path)

    # Combine the input with the main path via addition
    x = Add()([x, main_path])

    # Multiply the combined input with the refined channel weights
    x = Multiply()([x, main_path])

    # Flatten the output
    x = K.flatten(x)

    # Final classification layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.summary()