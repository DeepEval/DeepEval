import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the input images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Set up the input shape for the model
    input_shape = (32, 32, 3)

    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layer with global average pooling
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool_layer = GlobalAveragePooling2D()(conv_layer)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(pool_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)

    # Reshape weights to match input shape
    reshaped_weights = Reshape((-1, dense2.shape[1] * dense2.shape[2] * dense2.shape[3]))(dense2)

    # Element-wise multiplication with the input feature map
    output = tf.keras.backend.batch_dot(reshaped_weights, input_layer)

    # Flatten and output layer
    flat_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flat_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])