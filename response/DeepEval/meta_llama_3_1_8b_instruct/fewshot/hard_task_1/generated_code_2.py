import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Add, Dense, Reshape, Multiply, Lambda
from keras_initializer import initializers

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = GlobalAveragePooling2D()(conv_initial)
        path1 = Dense(32, activation='relu')(path1)
        path1 = Dense(10, activation='relu')(path1)
        path2 = GlobalMaxPooling2D()(conv_initial)
        path2 = Dense(32, activation='relu')(path2)
        path2 = Dense(10, activation='relu')(path2)
        attention = Dense(10, activation='sigmoid')(Add()([path1, path2]))
        attention = Reshape(target_shape=(1, 1, 10))(attention)
        features = Multiply()([conv_initial, attention])
        return features

    def block_2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        max_pool = GlobalMaxPooling2D()(avg_pool)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(avg_pool)
        max_pool = GlobalMaxPooling2D()(avg_pool)
        concat = Concatenate()([max_pool, avg_pool])
        normalized = Dense(10, activation='sigmoid')(Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat))
        normalized = Reshape(target_shape=(1, 1, 10))(normalized)
        features = Multiply()([input_tensor, normalized])
        return features

    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    conv_final = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    add = Add()([block1_output, conv_final])
    flatten = Flatten()(add)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model