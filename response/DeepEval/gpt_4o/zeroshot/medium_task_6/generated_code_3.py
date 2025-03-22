from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Softmax
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    initial_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)

    # First parallel block
    block1_conv = Conv2D(32, (3, 3), padding='same')(initial_conv)
    block1_bn = BatchNormalization()(block1_conv)
    block1_act = ReLU()(block1_bn)

    # Second parallel block
    block2_conv = Conv2D(32, (3, 3), padding='same')(initial_conv)
    block2_bn = BatchNormalization()(block2_conv)
    block2_act = ReLU()(block2_bn)

    # Third parallel block
    block3_conv = Conv2D(32, (3, 3), padding='same')(initial_conv)
    block3_bn = BatchNormalization()(block3_conv)
    block3_act = ReLU()(block3_bn)

    # Add outputs of the blocks to the initial convolution
    added = Add()([initial_conv, block1_act, block2_act, block3_act])

    # Flatten the added output
    flat = Flatten()(added)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flat)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer with Softmax activation
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of using the model
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Summary of the model
model.summary()

# You can now train the model using model.fit(), etc.
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))