import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Block 2
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    batch_norm2 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)

    # Block 3
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    batch_norm3 = BatchNormalization()(conv5)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm3)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)

    # Parallel branch
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm4 = BatchNormalization()(conv7)
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm4)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv8)

    # Aggregate outputs
    outputs = Concatenate()([max_pooling1, max_pooling2, max_pooling3, max_pooling4])

    # Flatten and fully connected layers
    flatten = Flatten()(outputs)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model