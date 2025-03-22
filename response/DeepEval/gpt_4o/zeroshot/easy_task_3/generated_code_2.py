from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for MNIST images (28x28 pixels, 1 channel)
    inputs = Input(shape=(28, 28, 1))

    # First block: 2 convolutions + max pooling
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block: 2 convolutions + max pooling
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third block: 3 convolutions + max pooling
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fourth block: 3 convolutions + max pooling
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(units=512, activation='relu')(x)
    x = Dense(units=256, activation='relu')(x)
    outputs = Dense(units=10, activation='softmax')(x)  # Output layer for 10 classes

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model