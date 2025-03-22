import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Concatenate
from keras.layers import Conv2DTranspose, LeakyReLU

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Reduce dimensionality with 1x1 convolution
    x = Conv2D(64, (1, 1), activation='relu')(inputs)

    # 3x3 convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(x)

    # Restore dimensionality with another 1x1 convolution
    x = Conv2D(64, (1, 1), activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer with 10 neurons for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Get the constructed model
model = dl_model()
model.summary()