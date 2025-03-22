import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define input shape
    input_shape = (32, 32, 3)

    # Input layer
    input_layer = Input(shape=input_shape)

    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)

    # Define three parallel blocks
    def block(input_tensor):
        # Batch normalization and ReLU activation
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.layers.Activation('relu')(batch_norm)

        # Max pooling
        pool = MaxPooling2D(pool_size=(2, 2))(relu)

        return pool

    # Apply the block three times
    block1_output = block(conv1)
    block2_output = block(block1_output)
    block3_output = block(block2_output)

    # Concatenate the outputs of the blocks
    concat = Concatenate()([conv1, block1_output, block2_output, block3_output])

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)