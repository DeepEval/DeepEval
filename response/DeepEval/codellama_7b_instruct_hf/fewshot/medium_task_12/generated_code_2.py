import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    batch_norm2 = BatchNormalization()(conv2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(batch_norm2)

    # Block 3
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    batch_norm3 = BatchNormalization()(conv3)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(batch_norm3)

    # Merge outputs of all blocks
    block_output = Concatenate()([max_pool1, max_pool2, max_pool3])

    # Flatten and fully connected layers
    flatten = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define and return model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model