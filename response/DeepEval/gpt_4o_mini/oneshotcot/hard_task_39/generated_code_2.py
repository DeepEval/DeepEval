import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three max pooling layers with varying scales
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling result
    flattened1 = Flatten()(pooling1)
    flattened2 = Flatten()(pooling2)
    flattened3 = Flatten()(pooling3)

    # Concatenate the flattened results
    block1_output = Concatenate()([flattened1, flattened2, flattened3])
    
    # Fully connected layer and reshape to prepare for Block 2
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_output = Reshape(target_shape=(1, 1, 128))(dense1)

    # Block 2: Multiple branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate all paths
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the result and add a dense layer for classification
    flat_block2_output = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flat_block2_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model