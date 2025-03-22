import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', depthwise=True)(conv1)

    # 1x1 convolutional layer with stride 2 to reduce dimensionality
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), activation='relu')(conv2)

    # MaxPooling2D with a pool size of 2
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Concatenate the outputs of the parallel paths
    concat = Concatenate(axis=-1)([pool, conv2, conv1, input_layer])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flat = Flatten()(batch_norm)

    # Fully connected layer for classification
    dense = Dense(units=128, activation='relu')(flat)
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Return the constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)