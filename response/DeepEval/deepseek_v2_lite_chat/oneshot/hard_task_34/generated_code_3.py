import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize inputs from 0-255 to 0-1
    x_train /= 255
    x_test /= 255

    # Define the main path
    input_layer = keras.Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm1)

    def block(input_tensor):
        conv = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm)
        return pool

    pool1_output = block(pool1)
    for _ in range(3):  # Repeat the block three times
        pool1_output = block(pool1_output)

    # Define the branch path
    branch_input = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(pool1_output)

    # Concatenate the main and branch paths
    concat = Concatenate()([pool1_output, branch_input])

    # Add a few more layers
    batch_norm2 = BatchNormalization()(concat)
    flat = Flatten()(batch_norm2)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train, x_train], y_train, epochs=10, validation_data=([x_test, x_test], y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test, x_test], y_test, verbose=2)
print('\nTest accuracy:', test_acc)