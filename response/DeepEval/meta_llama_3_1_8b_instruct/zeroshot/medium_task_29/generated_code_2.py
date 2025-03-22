# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to create the deep learning model
def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Convolutional layer with max pooling (1x1)
    x = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
    x = MaxPooling2D((1, 1), strides=1, padding='valid')(x)
    x = MaxPooling2D((1, 1), strides=1, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)

    # Convolutional layer with max pooling (2x2)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    # Convolutional layer with max pooling (4x4)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((4, 4), strides=4, padding='same')(x)
    x = MaxPooling2D((4, 4), strides=4, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # Concatenate the feature vectors
    x = concatenate([x, x, x])

    # Two fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model