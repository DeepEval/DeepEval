import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, Flatten, Dense, Multiply, Reshape, Add

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: First block - Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Step 4: Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Step 5: Main path - Add input to the block with the output of the main path
    main_path = Concatenate()([input_layer, avg_pool])

    # Step 6: Second block - Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(main_path)

    # Step 7: Reshape layer
    reshape_layer = Reshape((-1, 64))(global_avg_pool)

    # Step 8: Fully connected layer
    fc1 = Dense(units=64, activation='relu')(reshape_layer)

    # Step 9: Fully connected layer
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Step 10: Multiply layer
    multiply_layer = Multiply()([fc2, reshape_layer])

    # Step 11: Flatten layer
    flatten_layer = Flatten()(multiply_layer)

    # Step 12: Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model