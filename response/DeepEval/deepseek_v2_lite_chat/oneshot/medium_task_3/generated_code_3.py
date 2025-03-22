import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    # Define the model
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)

    # Combine outputs
    concat = Add()([conv1, conv2, maxpool1, maxpool2])

    # Flatten and fully connected layers
    flat = Flatten()(concat)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)