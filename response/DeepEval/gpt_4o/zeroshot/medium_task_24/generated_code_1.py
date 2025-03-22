from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(64, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: <max pooling>
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch3 = Dropout(0.3)(branch3)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated outputs
    x = Flatten()(concatenated)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to the fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model with CIFAR-10 data
if __name__ == '__main__':
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create the model
    model = dl_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))