import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 64))

    # Step 1: Compress the input channels with a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Expand features through parallel convolutional layers
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

    # Step 3: Concatenate the results from the parallel paths
    concatenated = Concatenate()([path1, path2])

    # Step 4: Flatten the output feature map into a one-dimensional vector
    flatten_layer = Flatten()(concatenated)

    # Step 5: Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model