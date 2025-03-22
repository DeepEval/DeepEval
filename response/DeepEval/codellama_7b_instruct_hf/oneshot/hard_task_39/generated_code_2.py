import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv2)
    flatten1 = Flatten()(maxpool2)

    # Block 2
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten1)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(conv5)
    flatten2 = Flatten()(maxpool3)

    # Concatenate the outputs of Block 1 and Block 2
    concatenated = Concatenate()([flatten1, flatten2])

    # Batch normalization and fully connected layers
    batch_norm = BatchNormalization()(concatenated)
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model