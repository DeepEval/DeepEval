import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Define the branch path
    global_avg_pool = GlobalAveragePooling2D()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=10, activation='relu')(dense1)
    channel_weights = Dense(units=128 * 4 * 4, activation='softmax')(dense2)
    channel_weights = Reshape((4, 4, 128))(channel_weights)
    channel_weights = Multiply()([channel_weights, conv3])

    # Add the outputs from both paths
    added_output = Concatenate()([max_pooling, channel_weights])
    dense3 = Dense(units=128, activation='relu')(added_output)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model