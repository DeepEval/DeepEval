from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class labels to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Create the initial convolution layer
    initial_conv = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x_train)

    # Create the three parallel blocks
    block_outputs = []
    for i in range(3):
        # Create a convolutional layer
        conv = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(initial_conv)

        # Create a batch normalization layer
        bn = layers.BatchNormalization()(conv)

        # Create a ReLU activation layer
        relu = layers.ReLU()(bn)

        # Append the block output to the list
        block_outputs.append(relu)

    # Add the block outputs to the initial convolution's output
    added_output = layers.Add()(initial_conv, block_outputs)

    # Flatten the added output
    flattened_output = layers.Flatten()(added_output)

    # Create the fully connected layers
    fc1 = layers.Dense(units=128, activation='relu')(flattened_output)
    fc2 = layers.Dense(units=10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=x_train, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Print the model summary
model = dl_model()
model.summary()