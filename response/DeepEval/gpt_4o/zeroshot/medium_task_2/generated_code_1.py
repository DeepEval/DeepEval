from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    # Branch path
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_layer)

    # Combine features from both paths
    combined = concatenate([x1, x2])

    # Flatten the combined features
    flat = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the model with CIFAR-10
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = dl_model()
    model.summary()
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))