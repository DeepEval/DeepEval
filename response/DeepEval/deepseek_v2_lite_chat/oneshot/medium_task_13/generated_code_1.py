import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv2)

    # Concatenate the outputs of the convolutional layers
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flat = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model