import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense
from keras.models import Model



def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize data
    x_train /= 255
    x_test /= 255

    # Define the main path
    input_main = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_main)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Define the branch path
    input_branch = Conv2D(64, (1, 1), activation='relu')(pool1)
    branch_conv1 = Conv2D(64, (3, 3), activation='relu')(input_branch)
    branch_pool1 = MaxPooling2D(pool_size=(2, 2))(branch_conv1)

    # Concatenate outputs from main and branch paths
    concat = Concatenate()([pool1, branch_pool1])

    # Fully connected layer followed by output layer
    fc1 = Flatten()(concat)
    output = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model

# Call the function and print the model
model = dl_model()