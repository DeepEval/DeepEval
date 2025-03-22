from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    x = Conv2D(32, (3, 3), padding='same', activation=None)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second block
    y = Conv2D(32, (3, 3), padding='same', activation=None)(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Concatenate()([x, y])
    
    # Third block
    z = Conv2D(32, (3, 3), padding='same', activation=None)(y)
    z = BatchNormalization()(z)
    z = ReLU()(z)
    z = Concatenate()([y, z])
    
    # Flatten and Fully Connected layers
    flat = Flatten()(z)
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Usage example
if __name__ == "__main__":
    model = dl_model()
    model.summary()

    # Prepare CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))