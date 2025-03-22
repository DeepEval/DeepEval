import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def block(x):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
        return conv2, maxpool2

    # Branch path
    branch_block = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_maxpool = MaxPooling2D(pool_size=(2, 2), padding='same')(branch_block)

    # Combine outputs from main path and branch path
    main_path_output, branch_output = block(input_layer)
    combined_output = Concatenate()([main_path_output, branch_output])
    combined_output = BatchNormalization()(combined_output)
    combined_output = Flatten()(combined_output)

    # Dense layers
    dense1 = Dense(units=256, activation='relu')(combined_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model