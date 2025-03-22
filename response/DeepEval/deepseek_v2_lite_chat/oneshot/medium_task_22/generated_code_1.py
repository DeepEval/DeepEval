import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Model architecture
    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)

    # Second branch with 1x1 convolution and two 3x3 convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)

    # Third branch with max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate the outputs from the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and pass through fully connected layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()

# Summary of the model
model.summary()