import keras
from keras.layers import Input, Conv2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Step 1: Compress the input channels with a 1x1 convolution
    compressed = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Expand the features through parallel convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

    # Step 3: Concatenate the results from both paths
    concatenated = Concatenate()([conv1, conv2])

    # Step 4: Flatten the output feature map into a one-dimensional vector
    flattened = Flatten()(concatenated)

    # Step 5: Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model