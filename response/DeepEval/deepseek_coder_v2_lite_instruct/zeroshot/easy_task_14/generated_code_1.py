import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = x_train.shape[1:]

    # Define the input layer
    inputs = Input(shape=input_shape)

    # First part of the model: Global Average Pooling
    x = GlobalAveragePooling2D()(inputs)

    # Second part of the model: Fully connected layers to generate weights
    x = Dense(128, activation='relu')(x)
    weights = Dense(input_shape[0], activation='sigmoid')(x)

    # Third part of the model: Reshape and element-wise multiplication
    reshaped_weights = Reshape((input_shape[0], 1, 1))(weights)
    x = Multiply()([inputs, reshaped_weights])

    # Fourth part of the model: Flatten and fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model

# Call the function to create and return the model
model = dl_model()