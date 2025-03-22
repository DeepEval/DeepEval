import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Create the main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Create the branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_input)

    # Combine paths using an addition operation
    combined = Add()([pool2, branch_conv])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flat = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flat)
    dense2 = Dense(128, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()