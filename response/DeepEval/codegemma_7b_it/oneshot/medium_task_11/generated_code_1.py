import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, concatenate, Activation, Multiply, AveragePooling2D, MaxPooling2D, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel-wise attention path
    avg_pool = GlobalAveragePooling2D()(conv)
    avg_pool = Dense(units=32, activation='relu')(avg_pool)
    max_pool = GlobalMaxPooling2D()(conv)
    max_pool = Dense(units=32, activation='relu')(max_pool)
    concat = concatenate([avg_pool, max_pool])
    attention_weights = Dense(units=32, activation='sigmoid')(concat)
    attention_weights = Flatten()(attention_weights)
    attention_weights = RepeatVector(32 * 32 * 32)(attention_weights)
    attention_weights = Reshape((32, 32, 32))(attention_weights)
    conv_attention = Multiply()([conv, attention_weights])

    # Spatial-wise attention path
    avg_pool_spatial = AveragePooling2D()(conv)
    max_pool_spatial = MaxPooling2D()(conv)
    concat_spatial = concatenate([avg_pool_spatial, max_pool_spatial])
    flatten_spatial = Flatten()(concat_spatial)
    dense_spatial = Dense(units=64, activation='relu')(flatten_spatial)
    dense_spatial = Dense(units=32, activation='relu')(dense_spatial)

    # Fusion and output
    concat_attention_spatial = concatenate([conv_attention, dense_spatial])
    flatten = Flatten()(concat_attention_spatial)
    dense = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model