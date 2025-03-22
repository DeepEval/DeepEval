import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 pooling layer
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # 2x2 pooling layer
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(pool1)

    # 4x4 pooling layer
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(pool2)

    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concat = Concatenate()([flatten1, flatten2, flatten3])

    # Batch normalization
    batch_norm = BatchNormalization()(concat)

    # Flatten the output of the batch normalization layer
    flat = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model