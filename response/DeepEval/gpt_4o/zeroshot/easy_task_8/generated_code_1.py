import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Define model input
    inputs = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Dropout(0.25)(x)  # Dropout after depthwise conv layer

    # 1x1 convolutional layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)  # Dropout after 1x1 conv layer

    # Flatten the output and add a fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()