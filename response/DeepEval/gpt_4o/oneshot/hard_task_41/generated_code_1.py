import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling with different scales and Dropout
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    path1_flat = Flatten()(path1)
    path1_drop = Dropout(rate=0.5)(path1_flat)

    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2_flat = Flatten()(path2)
    path2_drop = Dropout(rate=0.5)(path2_flat)

    path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    path3_flat = Flatten()(path3)
    path3_drop = Dropout(rate=0.5)(path3_flat)

    # Concatenate the paths
    block1_output = Concatenate()([path1_drop, path2_drop, path3_drop])

    # Fully connected layer and reshape operation between Block 1 and Block 2
    dense = Dense(units=7*7*16, activation='relu')(block1_output)  # Example: transforming to a 4D shape (7, 7, 16)
    reshaped = Reshape(target_shape=(7, 7, 16))(dense)

    # Block 2: Multiple branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)

    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    branch4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(reshaped)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Final fully connected layers for classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model