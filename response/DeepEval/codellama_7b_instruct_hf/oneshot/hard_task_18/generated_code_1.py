import keras
from keras.layers import Input, Conv2D, AvgPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define first block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AvgPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    concat = Concatenate()([conv1, conv2, avg_pool])
    batch_norm = BatchNormalization()(concat)

    # Define second block
    main_path = batch_norm
    global_avg_pool = GlobalAveragePooling2D()(main_path)
    fully_connected_1 = Dense(units=128, activation='relu')(global_avg_pool)
    fully_connected_2 = Dense(units=64, activation='relu')(fully_connected_1)
    channel_weights = fully_connected_2
    channel_weights = Reshape((32, 32))(channel_weights)
    channel_weights = Activation('softmax')(channel_weights)
    reshaped_input = Reshape((32, 32, 3))(input_layer)
    weighted_input = Multiply()([reshaped_input, channel_weights])
    flatten = Flatten()(weighted_input)
    fully_connected_3 = Dense(units=10, activation='softmax')(flatten)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=fully_connected_3)

    return model