import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Concatenate

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Global Average Pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layer 1
        dense1 = Dense(units=1024, activation='relu')(avg_pool)
        # Fully connected layer 2
        dense2 = Dense(units=1024, activation='relu')(dense1)
        return dense2

    block1_output = block1(input_tensor=input_layer)

    # Block 2
    def block2(input_tensor):
        # Convolutional layers
        conv1 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(conv1)
        # Max Pooling
        max_pool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv2)

        # Branch connection from Block 1
        branch_output = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(block1_output)

        # Concatenate the main path and the branch
        concat = Concatenate(axis=-1)([max_pool, branch_output])

        # Fully connected layers
        dense1 = Dense(units=2048, activation='relu')(concat)
        dense2 = Dense(units=1024, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    return block2(input_tensor=input_layer)

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)