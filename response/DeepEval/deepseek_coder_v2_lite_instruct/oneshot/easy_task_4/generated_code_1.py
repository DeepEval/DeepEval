import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block of convolutional layers followed by max pooling
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        return max_pooling

    first_block_output = first_block(input_layer)

    # Second block of convolutional layers followed by max pooling
    def second_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, max_pooling])
        return output_tensor

    second_block_output = second_block(first_block_output)

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model