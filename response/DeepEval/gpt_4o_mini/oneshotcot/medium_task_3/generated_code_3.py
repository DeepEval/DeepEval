import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: First block with Conv and MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Step 3: Second block with Conv and MaxPooling
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Step 4: Adjust dimensions for addition
    # Use a Conv2D layer to match the output dimensions with the input dimensions
    conv_res = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Step 5: Combine the outputs from the second block with the input through addition
    combined = Add()([max_pool2, conv_res])

    # Step 6: Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Step 7: Add a dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model