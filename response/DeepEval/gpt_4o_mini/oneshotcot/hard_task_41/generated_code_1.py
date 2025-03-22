import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))  # Input layer for MNIST images

    # Block 1: Three parallel paths with average pooling layers
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten and apply dropout
    flattened_path1 = Flatten()(path1)
    flattened_path2 = Flatten()(path2)
    flattened_path3 = Flatten()(path3)

    drop_path1 = Dropout(0.5)(flattened_path1)
    drop_path2 = Dropout(0.5)(flattened_path2)
    drop_path3 = Dropout(0.5)(flattened_path3)

    # Concatenate outputs of the three paths
    block1_output = Concatenate()([drop_path1, drop_path2, drop_path3])

    # Fully connected layer and reshape
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_output = Reshape((1, 1, 128))(dense1)

    # Block 2: Four branches for feature extraction
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)

    path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(reshaped_output)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate outputs of the four paths
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the output of Block 2
    flatten_block2 = Flatten()(block2_output)

    # Fully connected layers for classification
    dense2 = Dense(units=64, activation='relu')(flatten_block2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model