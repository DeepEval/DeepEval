import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Extract initial features
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)

    # Compress feature maps
    global_avg_pooling = GlobalAveragePooling2D()(batch_norm)
    dense1 = Dense(units=32, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape output and generate weighted feature maps
    output_dense = Dense(units=32)(dense2)
    output_reshape = Reshape((32, 1, 1))(output_dense)
    mul = Multiply()([output_reshape, input_layer])

    # Downsample and concatenate
    conv_down = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(mul)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_down)

    # Output layer
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model