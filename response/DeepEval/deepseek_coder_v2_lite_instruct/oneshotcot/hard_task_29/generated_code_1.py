import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Add convolutional layer to the main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Step 3: Add branch path directly to the input
    branch_path = input_layer

    # Step 4: Add addition layer to combine outputs from both paths
    combined_output = Add()([main_path, branch_path])

    # Second block
    # Step 5: Apply max pooling with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(combined_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(combined_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(combined_output)

    # Step 6: Flatten the results from the pooling layers
    flatten_layer = Flatten()(pool3)

    # Step 7: Add dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 8: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 9: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model