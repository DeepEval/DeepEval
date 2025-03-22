import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten outputs of path1, path2, and path3
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)

    # Concatenate outputs of paths
    block1_output = Concatenate()([flat_path1, flat_path2, flat_path3])

    # Fully connected layer between Block 1 and Block 2
    dense1 = Dense(units=128, activation='relu')(block1_output)

    # Reshape the output to fit Block 2
    reshaped_output = Reshape((1, 1, 128))(dense1)

    # Block 2
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate outputs of paths in Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the output from Block 2
    flatten_block2_output = Flatten()(block2_output)

    # Fully connected layers for classification
    dense2 = Dense(units=64, activation='relu')(flatten_block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model