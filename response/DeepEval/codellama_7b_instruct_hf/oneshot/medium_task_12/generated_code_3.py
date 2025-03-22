import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv2)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm2)

    # Second block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    batch_norm3 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm3)
    batch_norm4 = BatchNormalization()(conv4)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm4)

    # Third block
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    batch_norm5 = BatchNormalization()(conv5)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm5)
    batch_norm6 = BatchNormalization()(conv6)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm6)

    # Flatten layer
    flatten = Flatten()(max_pooling3)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model