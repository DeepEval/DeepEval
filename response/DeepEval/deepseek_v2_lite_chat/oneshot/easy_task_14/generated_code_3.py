import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the input image pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with global average pooling
    conv_layer = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to learn correlations
    dense1 = Dense(units=512, activation='relu')(conv_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)

    # Reshape weights to match the input shape
    reshaped_weights = Reshape((256, 16, 16))(dense2)

    # Element-wise multiplication with the input feature map
    multiplied_output = keras.backend.batch_dot(reshaped_weights, input_layer)

    # Flatten and final dense layer
    flatten = Flatten()(multiplied_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()