import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)

    # Normalize data
    x_train /= 255
    x_test /= 255

    # Parameters
    img_rows, img_cols, img_channels = 28, 28, 1
    num_classes = 10

    # Define the input layer
    input_layer = Input(shape=(img_rows, img_cols, img_channels))

    # Block for the first branch
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(conv2)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        return max_pool1

    branch1_output = block(input_tensor=input_layer)

    # Repeat the block for the second branch
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(conv2)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        return max_pool1

    branch2_output = block(input_tensor=input_layer)

    # Concatenate the outputs from both branches
    concat_layer = Concatenate(axis=-1)([branch1_output, branch2_output])

    # Add batch normalization and flatten the concatenated layer
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)

    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=num_classes, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)