import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Add, Activation
from keras.regularizers import l2
from keras.initializers import RandomNormal

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), kernel_regularizer=l2(0.01))(input_layer)

    def channel_attention(input_tensor):

        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=128, activation='relu')(path1)
        path1 = Dense(units=32, activation='relu')(path1)
        path1 = Dense(units=1, activation='sigmoid')(path1)

        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=128, activation='relu')(path2)
        path2 = Dense(units=32, activation='relu')(path2)
        path2 = Dense(units=1, activation='sigmoid')(path2)

        output_tensor = Add()([path1, path2])
        output_tensor = Activation('sigmoid')(output_tensor)

        return Multiply()([input_tensor, output_tensor])

    channel_features = channel_attention(conv)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(channel_features)

    fused_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Multiply()([channel_features, fused_features])

    batch_norm = BatchNormalization()(spatial_features)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model