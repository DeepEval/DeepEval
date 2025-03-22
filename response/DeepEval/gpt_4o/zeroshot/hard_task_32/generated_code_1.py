from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam

def dl_model():
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Specialized block as a function for reusability
    def specialized_block(x):
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        x = Dropout(0.3)(x)
        return x

    # Create three branches
    branch1 = specialized_block(inputs)
    branch2 = specialized_block(inputs)
    branch3 = specialized_block(inputs)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers
    x = Flatten()(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# To load the MNIST dataset and train the model
if __name__ == "__main__":
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Get model
    model = dl_model()

    # Train model
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

    # Evaluate model
    score = model.evaluate(x_test, y_test)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')