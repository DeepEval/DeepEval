import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pooling1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Main path with skip connection
    skip_connection = Add()([input_layer, avg_pooling1])

    # Second block
    global_avg_pooling = GlobalAveragePooling2D()(skip_connection)
    channel_weights = Dense(units=32, activation='relu')(global_avg_pooling)
    channel_weights = Dense(units=32, activation='sigmoid')(channel_weights)
    
    # Reshape to multiply with the main path
    reshaped_weights = Reshape((1, 1, 32))(channel_weights)

    # Multiply the weights with the skip connection
    weighted_output = keras.layers.multiply([skip_connection, reshaped_weights])

    # Flatten and classify
    flatten_layer = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model