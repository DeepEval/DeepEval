import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three max pooling layers with varying scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening the outputs of the max pooling layers
    flat1 = Flatten()(max_pool1)
    flat2 = Flatten()(max_pool2)
    flat3 = Flatten()(max_pool3)

    # Concatenating the flattened outputs
    block1_output = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer before reshaping
    dense_block1 = Dense(units=128, activation='relu')(block1_output)

    # Reshaping to a 4D tensor suitable for Block 2
    reshaped_block1 = Reshape((1, 1, 128))(dense_block1)

    # Block 2: Multiple branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_block1)

    # Concatenating the outputs of the branches
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flattening the output of Block 2
    flatten_output = Flatten()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model