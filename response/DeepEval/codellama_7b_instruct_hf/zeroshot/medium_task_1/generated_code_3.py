import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pooling = MaxPooling2D((2, 2))(conv2)
    flatten = Flatten()(pooling)
    dense1 = Dense(128, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model