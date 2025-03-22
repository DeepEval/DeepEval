import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    The model is structured into two blocks: The first block features both a main path and a branch path.
    The second block employs three max pooling layers with varying scales.
    After processing through both blocks, the model passes through two fully connected layers to produce the final classification result.
    
    Returns:
        model: The constructed deep learning model.
    """

    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Add convolutional layer as the main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add convolutional layer as the branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 4: Combine the outputs from both paths
    block_output = Add()([main_path, branch_path])

    # Step 5: Define the second block
    def block2(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    # Step 6: Apply the second block
    block_output = block2(block_output)

    # Step 7: Add batch normalization and flatten the result
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)

    # Step 8: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 9: Add another dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 10: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 11: Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model