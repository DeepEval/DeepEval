import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, AveragePooling2D

def dl_model():     
    # Input layer for MNIST images
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening the output of each path
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)

    # Applying Dropout for regularization
    drop_path1 = Dropout(0.5)(flat_path1)
    drop_path2 = Dropout(0.5)(flat_path2)
    drop_path3 = Dropout(0.5)(flat_path3)

    # Concatenating the outputs from the three paths
    concatenated_output = Concatenate()([drop_path1, drop_path2, drop_path3])

    # Fully connected layer
    fc_layer = Dense(128, activation='relu')(concatenated_output)

    # Reshape to prepare for Block 2
    reshaped_output = Reshape((4, 4, 8))(fc_layer)  # Note: Ensure the total dimensions match

    # Block 2
    path1_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path2_block2)
    path2_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2_block2)

    path3_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_block2)

    path4_block2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped_output)
    path4_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_block2)

    # Concatenating outputs from Block 2
    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Flattening the output for the final classification layers
    flat_block2_output = Flatten()(block2_output)

    # Final fully connected layers
    dense1 = Dense(128, activation='relu')(flat_block2_output)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model