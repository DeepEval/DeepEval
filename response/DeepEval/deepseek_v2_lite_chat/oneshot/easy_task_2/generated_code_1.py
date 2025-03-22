import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))

    # First feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)

    # Second feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv2)

    # Additional convolutional layers
    def block(input_tensor):
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv3)
        return avg_pool3

    conv3_output = block(avg_pool2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_output)
    avg_pool4 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv4)

    # Concatenate the feature maps from the convolutional layers
    def concatenate_layers(input_tensor1, input_tensor2):
        concat = Concatenate()(input_tensor1)
        return concat

    concatenated_output = concatenate_layers(avg_pool4, conv4)

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(batch_norm)

    # Two fully connected layers with dropout
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    drop1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(drop1)
    drop2 = Dropout(rate=0.5)(dense2)

    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model