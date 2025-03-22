import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Max Pooling Layers
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the pooling results
    flatten1 = Flatten()(pooling1)
    flatten2 = Flatten()(pooling2)
    flatten3 = Flatten()(pooling3)

    # Concatenate the flattened outputs
    block1_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer and reshape
    dense_block1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_output = Reshape((4, 4, 8))(dense_block1)  # Reshape to 4D tensor suitable for Block 2

    # Block 2
    # Branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate the outputs from the branches
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and output layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model