import keras
from keras.layers import Input, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, Conv2D, AveragePooling2D, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: Max Pooling 1x1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.2)(path1)

    # Path 2: Max Pooling 2x2
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.2)(path2)

    # Path 3: Max Pooling 4x4
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.2)(path3)

    # Concatenate the outputs of the three paths
    block1_output = Concatenate()([path1, path2, path3])
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(block1_output)

    # Reshape to 4D tensor for Block 2
    reshaped_output = Reshape((1, 1, 128))(dense1)

    # Block 2
    # Path 1: 1x1 Convolution
    path1_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)

    # Path 2: 1x1 Convolution -> 1x7 Convolution -> 7x1 Convolution
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path2_block2)
    path2_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2_block2)

    # Path 3: 1x1 Convolution -> 7x1 Convolution -> 1x7 Convolution (repeated)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_block2)

    # Path 4: Average Pooling + 1x1 Convolution
    path4_block2 = AveragePooling2D(pool_size=(2, 2))(reshaped_output)
    path4_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_block2)

    # Concatenate outputs of Block 2
    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Flatten and apply dense layers
    flatten_layer = Flatten()(block2_output)
    dense2 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model