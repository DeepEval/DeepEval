import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape the data to match the input shape of the model
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the paths for the two branches
    path1 = Input(shape=(28, 28, 1))
    path2 = Input(shape=(28, 28, 1))

    # Path 1: Convolutional layers followed by max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(path1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Path 2: Directly from input
    direct_output = path2

    # Combine the two paths using an addition operation
    combined = Add()([pool1, direct_output])

    # Flatten and fully connected layers
    flatten = Flatten()(combined)
    dense = Dense(128, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=[path1, path2], outputs=output)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Create and train the model
model = dl_model()
model.fit([x_train, x_train], y_train, epochs=5, batch_size=128)

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test, x_test], y_test, verbose=2)
print('\nTest accuracy:', test_acc)