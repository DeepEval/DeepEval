import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D, BatchNormalization

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with average pooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)

    # Concatenate the flattened outputs
    block1_output = Concatenate()([flat_path1, flat_path2, flat_path3])
    
    # Fully connected layer after Block 1
    dense1 = Dense(units=128, activation='relu')(block1_output)
    
    # Reshape the output to prepare for Block 2
    reshaped_output = Reshape((1, 1, 128))(dense1)

    # Block 2: Three branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate the outputs from Block 2
    block2_output = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    block2_flattened = Flatten()(block2_output)

    # Fully connected layers for classification
    dense2 = Dense(units=64, activation='relu')(block2_flattened)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model