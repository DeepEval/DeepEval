import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add convolutional layer (initial layer) to reduce dimensionality to 16
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Define a basic block
    def basic_block(input_tensor):
        # Main path
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        main_path = ReLU()(x)

        # Branch path (direct connection)
        branch = input_tensor

        # Feature fusion
        output_tensor = Add()([main_path, branch])
        return output_tensor

    # Apply the basic block twice
    block1 = basic_block(conv1)
    block2 = basic_block(block1)

    # Additional convolutional layer in the branch
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature fusion with the branch output
    combined = Add()([block2, branch_conv])

    # Step 4: Add average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1)(combined)

    # Step 5: Flatten the result
    flatten_layer = Flatten()(avg_pool)

    # Step 6: Add dense layer
    dense1 = Dense(units=64, activation='relu')(flatten_layer)

    # Step 7: Add final dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Step 8: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model