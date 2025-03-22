import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch Path
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    channel_weights = Dense(units=64 * 64 * 3, activation='linear')(dense2)
    reshaped_weights = Reshape((64, 64, 3))(channel_weights)

    # Element-wise Multiplication
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Concatenate outputs and final classification
    concat_layer = keras.layers.concatenate([pool, weighted_input], axis=3)
    dense4 = Dense(units=128, activation='relu')(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model