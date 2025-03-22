import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Flatten and concatenate the outputs of the three parallel paths
    flatten1 = Flatten()(max_pooling1)
    flatten2 = Flatten()(max_pooling2)
    flatten3 = Flatten()(max_pooling3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Add batch normalization and fully connected layers
    batch_normalized = BatchNormalization()(concatenated)
    dense1 = Dense(units=128, activation='relu')(batch_normalized)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Block 2
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(dense2)
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling4)
    max_pooling5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)

    conv6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(max_pooling5)
    max_pooling6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)

    # Flatten and concatenate the outputs of the four parallel paths
    flatten4 = Flatten()(max_pooling4)
    flatten5 = Flatten()(max_pooling5)
    flatten6 = Flatten()(max_pooling6)
    concatenated_block2 = Concatenate()([flatten4, flatten5, flatten6])

    # Add batch normalization and fully connected layers
    batch_normalized_block2 = BatchNormalization()(concatenated_block2)
    dense3 = Dense(units=128, activation='relu')(batch_normalized_block2)
    dense4 = Dense(units=64, activation='relu')(dense3)

    # Output layer
    output = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model