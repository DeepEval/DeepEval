import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First stage of convolution and max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second stage of convolution and max pooling
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)

    # Flatten and pass through three fully connected layers
    flat1 = Flatten()(pool2)
    dense1 = Dense(512, activation='relu')(flat1)
    dropout1 = Dense(512, activation='relu')(dense1)
    dropout2 = Dense(512, activation='relu')(dropout1)
    output = Dense(10, activation='softmax')(dropout2)

    # Model architecture
    model = Model(inputs=inputs, outputs=output)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()