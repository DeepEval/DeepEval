import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Main path with two consecutive 3x3 convolutional layers
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path_conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(main_path_conv1)
    
    # Step 3: Max pooling layer
    main_path_pool = MaxPooling2D(pool_size=(2, 2))(main_path_conv2)

    # Step 4: Branch path with a single 5x5 convolutional layer
    branch_path_conv = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Step 5: Combine the outputs of the main path and branch path
    combined = Concatenate()([main_path_pool, branch_path_conv])

    # Step 6: Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Step 7: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Step 8: Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model