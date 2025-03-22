from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Dropout(0.3)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu', padding='same')(branch4)
    branch4 = Dropout(0.3)(branch4)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and add fully connected layers
    x = Flatten()(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Get the model
    model = dl_model()

    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))