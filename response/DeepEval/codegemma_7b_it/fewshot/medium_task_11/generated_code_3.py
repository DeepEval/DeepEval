import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Lambda, Multiply, concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def channel_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)

        avg_dense1 = Dense(units=32, activation='relu')(avg_pool)
        avg_dense2 = Dense(units=1, activation='sigmoid')(avg_dense1)

        max_dense1 = Dense(units=32, activation='relu')(max_pool)
        max_dense2 = Dense(units=1, activation='sigmoid')(max_dense1)

        output_tensor = Multiply()([input_tensor, avg_dense2, max_dense2])
        return output_tensor

    channel_output = channel_path(input_tensor=initial_conv)

    def spatial_path(input_tensor):
        avg_pool = AveragePooling2D()(input_tensor)
        max_pool = MaxPooling2D()(input_tensor)

        flatten_avg = Flatten()(avg_pool)
        flatten_max = Flatten()(max_pool)

        dense_avg = Dense(units=128, activation='relu')(flatten_avg)
        dense_max = Dense(units=128, activation='relu')(flatten_max)

        concat = concatenate([dense_avg, dense_max], axis=-1)
        output_tensor = Dense(units=1, activation='sigmoid')(concat)

        return output_tensor

    spatial_output = spatial_path(input_tensor=initial_conv)

    concat_features = concatenate([channel_output, spatial_output], axis=-1)

    flatten = Flatten()(concat_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model