from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, Add, Flatten, Dense, Input
from keras.layers import SeparableConv2D

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    input_shape = (28, 28, 1)

    # Convert labels to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Define the model architecture
    input_layer = Input(shape=input_shape)

    # Increase dimensionality
    x = Conv2D(16, (1, 1), activation='relu')(input_layer)

    # Depthwise separable convolutional layer
    x = SeparableConv2D(32, (3, 3), activation='relu')(x)

    # Reduce dimensionality
    x = Conv2D(8, (1, 1), activation='relu')(x)

    # Add the original input layer to the processed output
    added = Add()([x, input_layer])

    # Flatten the output
    x = Flatten()(added)

    # Fully connected layer
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()