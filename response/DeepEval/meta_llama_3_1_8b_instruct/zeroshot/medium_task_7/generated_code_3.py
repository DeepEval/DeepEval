from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class labels to categorical labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the model
    def create_model():
        # Input layer
        inputs = layers.Input(shape=(32, 32, 3))

        # Path 1: Three sequential convolutional layers
        conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
        conv1 = layers.Conv2D(128, (3, 3), activation='relu')(conv1)
        path1 = layers.Add()([conv1, conv1, conv1])  # Add the outputs of the three convolutional layers

        # Path 2: Directly processing the input with a separate convolutional layer
        conv2 = layers.Conv2D(128, (3, 3), activation='relu')(inputs)

        # Merge the outputs of both paths
        merged = layers.Concatenate()([path1, conv2])

        # Flatten the merged output
        flattened = layers.Flatten()(merged)

        # Two fully connected layers for classification
        fc1 = layers.Dense(128, activation='relu')(flattened)
        outputs = layers.Dense(10, activation='softmax')(fc1)

        # Define the model
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Create the model
    model = create_model()

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model