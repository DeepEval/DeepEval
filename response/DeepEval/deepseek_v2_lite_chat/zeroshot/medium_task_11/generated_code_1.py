import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Flatten
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Constants
    IMG_ROWS, IMG_COLS, CHANNELS = 32, 32, 3
    NUM_CLASSES = 10
    NUM_EPOCHS = 20
    BATCH_SIZE = 128

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Model parameters
    input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)
    model = tf.keras.models.Sequential([
        # Convolutional layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Path 1: Global Average Pooling
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        
        # Path 2: Global Max Pooling
        GlobalMaxPooling2D(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        
        # Combining the outputs from both paths
        Concatenate(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()