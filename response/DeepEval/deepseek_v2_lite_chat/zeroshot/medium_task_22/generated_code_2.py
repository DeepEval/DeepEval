import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model architecture
def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)

    # Branch 2: 1x1 conv -> 2x3 conv -> 3x3 conv
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)

    # Branch 3: max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate feature maps from different branches
    concat = concatenate([branch1, branch2, branch3])

    # Flatten and pass through fully connected layers
    flat = Flatten()(concat)
    fc1 = Dense(512, activation='relu')(flat)
    fc2 = Dense(10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()