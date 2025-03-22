import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Adjust feature dimensionality to 16 using a convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Basic block
    def block(input_tensor):
        path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = BatchNormalization()(path1)
        path3 = Activation('relu')(path2)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    # First level of residual connection structure
    block_output = block(input_tensor=conv)

    # Second level of residual connection structure
    for i in range(2):
        # Main path
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)
        main_path = BatchNormalization()(main_path)
        main_path = Activation('relu')(main_path)

        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Addition operation
        output_tensor = Concatenate()([main_path, branch])

        # Residual block
        block_output = block(input_tensor=output_tensor)

    # Third level of residual connection structure
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    output_tensor = Concatenate()([block_output, global_branch])

    # Average pooling
    output_tensor = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(output_tensor)

    # Flatten
    output_tensor = Flatten()(output_tensor)

    # Fully connected layers
    output_tensor = Dense(units=128, activation='relu')(output_tensor)
    output_tensor = Dense(units=64, activation='relu')(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model