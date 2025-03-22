import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Step 2: First convolutional block
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_main)

    # Step 3: Second convolutional block
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1_main)
    max_pool2_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_main)

    # Branch path
    # Step 4: Convolutional block in branch
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)

    # Step 5: Combine outputs from main path and branch path
    combined_output = Add()([max_pool2_main, max_pool_branch])

    # Step 6: Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Step 7: Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 8: Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model