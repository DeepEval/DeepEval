import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, BatchNormalization, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 image size

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = GlobalAveragePooling2D()(conv1)
    max_pool1 = GlobalMaxPooling2D()(conv1)
    fc1 = Dense(units=128, activation='relu')(avg_pool1)
    fc2 = Dense(units=128, activation='relu')(max_pool1)
    channel_attention = Concatenate()([fc1, fc2])
    channel_attention = Dense(units=32, activation='relu')(channel_attention)
    channel_attention = Dense(units=32, activation='sigmoid')(channel_attention)
    channels = keras.backend.expand_dims(channel_attention, axis=-1)
    channels = keras.backend.expand_dims(channels, axis=-1)
    conv1_attention = keras.backend.multiply([conv1, channels])

    # Block 2
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    concat = Concatenate()([conv2, conv3])
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat)

    # Final classification
    flatten = Flatten()(conv4)
    dense = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model