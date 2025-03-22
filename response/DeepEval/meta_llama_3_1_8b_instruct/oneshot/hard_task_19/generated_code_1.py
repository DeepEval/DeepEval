import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Multiply, Reshape, Concatenate

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch Path
    global_avg_pooling = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Get the channel weights
    channel_weights = Dense(units=128, activation='sigmoid')(dense2)
    channel_weights = Reshape((128,))(channel_weights)

    # Multiply the channel weights with the input
    multiplied_input = Multiply()([conv3, channel_weights])

    # Concatenate the main path and branch path
    concatenated_output = Concatenate()([max_pooling, multiplied_input])

    # Apply two additional fully connected layers for classification
    flatten_layer = Flatten()(concatenated_output)
    dense3 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model