import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # First block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

        # Second block
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)

        return maxpool2

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First block
    block1_output = block(conv1)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten1 = Flatten()(batch_norm1)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()
model.summary()