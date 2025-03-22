# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Define the three blocks of the model
    def block(x):
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    x = block(inputs)
    x = block(x)
    x = block(x)

    # Concatenate the output of the three blocks along the channel dimension
    x = Concatenate()([x, x, x])

    # Flatten the output
    x = Flatten()(x)

    # Add two fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model