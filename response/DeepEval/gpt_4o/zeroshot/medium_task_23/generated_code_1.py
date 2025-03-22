from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 Convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(64, (1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(64, (7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by two sets of 1x7 and 7x1 Convolutions
    path3 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path3 = Conv2D(64, (1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(64, (7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(64, (1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(64, (7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(64, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of the four paths
    concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten = Flatten()(concatenated)

    # Fully connected layer for classification
    dense = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model

# Load CIFAR-10 data to check the model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()