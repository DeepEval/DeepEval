import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # 1. Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 2. First Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # 3. Max Pooling layer
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second block
    # Define a block function
    def block(input_tensor):
        # 4.1. First Convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 4.2. Second Convolutional layer
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        # 4.3. Third Convolutional layer
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        # 4.4. Max Pooling layer
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
        # Merge all paths
        output_tensor = Concatenate()([maxpool1, maxpool2])
        return output_tensor

    block_output = block(maxpool1)
    # 5. Batch normalization
    batch_norm = BatchNormalization()(block_output)
    # 6. Flatten layer
    flatten_layer = Flatten()(batch_norm)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Call the function to create the model
model = dl_model()
model.summary()