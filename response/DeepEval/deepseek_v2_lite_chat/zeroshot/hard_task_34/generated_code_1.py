import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Define the model
def dl_model():
    # Main path
    input1 = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input1)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Add()([x, input1])

    # Branch path
    input2 = Conv2D(64, (3, 3), activation='relu')(input1)

    # Concatenate and flatten
    x = Add()([x, input2])
    x = Flatten()(x)

    # Fully connected layer
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input1, input2], outputs=[output])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()
model.summary()