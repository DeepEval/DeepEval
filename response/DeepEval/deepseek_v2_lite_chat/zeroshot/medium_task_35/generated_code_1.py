import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential


def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Model architecture
    model = Sequential([
        # Stage 1: Convolution and Max Pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        # Stage 2: Additional Convolution and Dropout
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Stage 3: Downsampling to 7x7 feature map
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Stage 4: Reduce to 4x4 feature map
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Stage 5: Spatial pyramid pooling, then 1x1 convolution
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Stage 6: Skip connections with upsampling
        Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),
        Concatenate(),
        Conv2D(128, (3, 3), activation='relu'),
        Dropout(0.5),

        # Stage 7: Further upsampling and skip connections
        Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
        Concatenate(),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.5),

        # Stage 8: Output layer
        Conv2D(num_classes, (1, 1), activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Return the compiled model
    return model

# Instantiate and return the model
model = dl_model()