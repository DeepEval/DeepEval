from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Average pooling layers with different window sizes and strides
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)

    # Flatten each pooled feature map
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate flattened features
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    x = Dense(256, activation='relu')(concatenated)
    x = Dense(128, activation='relu')(x)

    # Output layer for 10 classes (CIFAR-10)
    output_layer = Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build the model
    model = dl_model()

    # Summary of the model
    model.summary()

    # Train the model (example)
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))