import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten and apply Dropout
    flat1 = Flatten()(path1)
    drop1 = Dropout(0.5)(flat1)
    
    flat2 = Flatten()(path2)
    drop2 = Dropout(0.5)(flat2)
    
    flat3 = Flatten()(path3)
    drop3 = Dropout(0.5)(flat3)

    # Concatenate outputs of all paths
    block1_output = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer and reshaping
    dense_block1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_output = Reshape((4, 4, 8))(dense_block1)  # Reshape to (4, 4, 8) for next block

    # Block 2
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path1)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(reshaped_output)

    path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshaped_output)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)

    # Concatenate outputs of all branches in block 2
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the output of block 2 and apply fully connected layers for classification
    flat_block2_output = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flat_block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model