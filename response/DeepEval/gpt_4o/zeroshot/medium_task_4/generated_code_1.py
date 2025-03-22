from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)

    # Path 2: Single convolutional layer
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

    # Combine the features from both paths
    combined = Add()([x1, x2])

    # Flatten the combined features
    flat = Flatten()(combined)

    # Fully connected layer mapping to 10 classes
    output_layer = Dense(10, activation='softmax')(flat)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()