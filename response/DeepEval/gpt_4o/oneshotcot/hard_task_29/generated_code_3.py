import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    # Main path
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    
    # Branch path (identity connection)
    branch_path = input_layer

    # Combine outputs of main and branch paths
    first_block_output = Add()([main_conv2, branch_path])

    # Second Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(first_block_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(first_block_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(first_block_output)

    # Flatten and concatenate the outputs of the pooling layers
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    second_block_output = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model