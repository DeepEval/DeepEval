from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer - MNIST images are 28x28 pixels in grayscale
    input_layer = Input(shape=(28, 28, 1))

    # First Convolutional Layer followed by Max Pooling
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    # Second Convolutional Layer followed by Max Pooling
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Third Convolutional Layer
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # Flattening the 3D feature maps to 1D feature vectors
    x = Flatten()(x)

    # First Fully Connected Layer
    x = Dense(64, activation='relu')(x)

    # Second Fully Connected Layer
    x = Dense(64, activation='relu')(x)

    # Output Layer with 10 units for the 10 classes (digits 0-9)
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model