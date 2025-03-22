from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the two blocks
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max pooling with varying scales
    block1 = Conv2D(64, kernel_size=1, activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1)
    block1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(block1)
    block1 = Flatten()(block1)

    # Block 2: Concatenation of flattened max pooling results
    block2 = Flatten()(block1)

    # Converting the output of Block 1 to be 4D tensor
    reshape = Reshape((-1, 4))(block1)

    # Second block with multiple branches
    branch1 = Conv2D(64, kernel_size=1, activation='relu')(input_layer)
    branch2 = Conv2D(64, kernel_size=3, activation='relu')(branch1)
    branch3 = Conv2D(64, kernel_size=5, activation='relu')(branch2)
    branch4 = MaxPooling2D(pool_size=(3, 3))(branch3)
    branch1 = Flatten()(branch1)
    branch2 = Flatten()(branch2)
    branch3 = Flatten()(branch3)
    branch4 = Flatten()(branch4)

    # Concatenate the outputs from all branches
    output = Concatenate()([branch1, branch2, branch3, branch4])

    # Final dense layer for classification
    dense = Dense(10, activation='softmax')(output)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and return the model
model = dl_model()