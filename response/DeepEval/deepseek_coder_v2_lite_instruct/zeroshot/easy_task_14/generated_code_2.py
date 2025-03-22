import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Add convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(x)

    # Fully connected layers to generate weights
    weights = Dense(32, activation='relu')(gap)
    weights = Dense(32, activation='relu')(weights)
    weights = Dense(32, activation='sigmoid')(weights)

    # Reshape weights to align with the input shape
    weights = tf.reshape(weights, (-1, 32, 1, 1))

    # Multiply element-wise with the input feature map
    x = Multiply()([x, weights])

    # Flatten the result
    x = Flatten()(x)

    # Fully connected layer for output
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model

# Call the function to create the model
model = dl_model()