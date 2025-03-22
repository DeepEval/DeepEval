import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(conv)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Global max pooling
    global_max_pool = GlobalMaxPooling2D()(conv)

    # Fully connected layers
    dense3 = Dense(units=128, activation='relu')(global_max_pool)
    dense4 = Dense(units=64, activation='relu')(dense3)

    # Channel attention weights
    attention_weights = Add()([dense2, dense4])
    attention_weights = Activation('softmax')(attention_weights)

    # Element-wise multiplication
    channel_attention = Concatenate()([conv, attention_weights])

    # Separate average and max pooling operations
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_attention)

    # Concatenate spatial features
    concat_layer = Concatenate()([avg_pool, max_pool])

    # Flatten and fully connected layers
    flatten = Flatten()(concat_layer)
    dense5 = Dense(units=128, activation='relu')(flatten)
    dense6 = Dense(units=64, activation='relu')(dense5)
    output_layer = Dense(units=10, activation='softmax')(dense6)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model