import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three max pooling layers with varying scales
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling result
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)

    # Concatenate flattened outputs
    concatenated_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer to convert into a suitable shape for Block 2
    dense_block1 = Dense(units=128, activation='relu')(concatenated_output)
    
    # Reshape output to 4D tensor suitable for Block 2
    reshaped_output = Reshape((1, 1, 128))(dense_block1)  # Reshaping to (1, 1, 128) to fit into conv layers

    # Block 2: Multiple branches for feature extraction
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten_block2_output = Flatten()(block2_output)

    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_block2_output)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model