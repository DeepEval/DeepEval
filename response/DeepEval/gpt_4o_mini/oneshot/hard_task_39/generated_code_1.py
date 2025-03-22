import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three MaxPooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling result
    flat_pool1 = Flatten()(pool1)
    flat_pool2 = Flatten()(pool2)
    flat_pool3 = Flatten()(pool3)

    # Concatenate flattened outputs
    concat_block1 = Concatenate()([flat_pool1, flat_pool2, flat_pool3])

    # Fully connected layer and reshape for Block 2
    dense_block1 = Dense(units=128, activation='relu')(concat_block1)
    reshaped_block1 = Reshape((1, 1, 128))(dense_block1)  # Reshaping to (1, 1, 128) for Block 2

    # Block 2: Multiple branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_block1)

    # Concatenate outputs from all branches
    concat_block2 = Concatenate()([path1, path2, path3, path4])

    # Flatten and classify
    flatten_block2 = Flatten()(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(flatten_block2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model