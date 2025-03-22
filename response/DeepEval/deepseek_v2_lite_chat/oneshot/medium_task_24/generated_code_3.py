import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D, Conv2D as Conv2D_custom, AveragePooling2D
from keras.models import Model
from keras.datasets import cifar10
import numpy as np

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Three branches for enhanced feature extraction
    def branch1(input_tensor):
        conv1 = Conv2D_custom(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D_custom(filters=32, kernel_size=(3, 3), padding='same')(conv1)
        dropout1 = keras.layers.Dropout(0.5)(conv2)
        return dropout1

    def branch2(input_tensor):
        conv1 = Conv2D_custom(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D_custom(filters=64, kernel_size=(7, 1), padding='valid')(conv1)
        conv3 = Conv2D_custom(filters=64, kernel_size=(1, 7), padding='same')(conv2)
        conv4 = Conv2D_custom(filters=64, kernel_size=(3, 3), padding='valid')(conv1)
        conv5 = Conv2D_custom(filters=64, kernel_size=(7, 1), padding='valid')(conv2)
        avg_pool = AveragePooling2D(pool_size=(1, 1))(input_tensor)
        dropout2 = keras.layers.Dropout(0.5)(avg_pool)
        return Concatenate()([dropout1, conv2, conv3, conv4, dropout2])

    def branch3(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(1, 3), strides=(1, 1))(input_tensor)
        return avg_pool

    # Process each branch
    branch1_output = branch1(input_layer)
    branch2_output = branch2(input_layer)
    branch3_output = branch3(input_layer)

    # Concatenate the outputs
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and display the model
model = dl_model()
model.summary()