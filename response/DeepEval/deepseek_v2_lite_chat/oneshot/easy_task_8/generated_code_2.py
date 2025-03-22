import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalize data
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Function to create model
def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable conv layer
    conv = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv = Conv2D(32, (3, 3), activation='relu', padding='same')(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)

    # 1x1 convolutional layer for feature extraction
    dense1 = Conv2D(64, (1, 1), activation='relu')(conv)

    # Dropout layer to prevent overfitting
    dense1 = Dropout(0.5)(dense1)

    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(dense1)
    dense2 = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])