from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import LayerNormalization
from keras.optimizers import Adam

def dl_model():
    # Define the input shape (assuming 32x32x3 images, as per CIFAR-10)
    input_shape = (32, 32, 3)
    num_classes = 10  # CIFAR-10 has 10 classes

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Set up the input layers
    inputs = Input(shape=input_shape)

    # Encoder layers
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output to prepare for the decoder
    x = Flatten()(x)

    # Decoder layers
    x = Dense(128)(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(64)(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(3 * 3 * 128, activation='relu')(x)

    # Reshape for output shape (32, 32, 128)
    x = Reshape((3, 3, 128))(x)

    # Concatenate features from different scales
    x = concatenate([x, x])  # Concatenate with the original input
    x = Conv2DTranspose(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=(3, 3), activation='relu')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()