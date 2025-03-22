import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Add, Lambda
from keras import backend as K
from tensorflow.keras import regularizers

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    initial_conv = BatchNormalization()(initial_conv)
    initial_conv ='relu')(initial_conv)

    def block1(input_tensor):

        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=128, activation='relu')(path1)
        path1 = Dense(units=64, activation='relu')(path1)
        path1 = Dense(units=10, activation='softmax')(path1)

        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=128, activation='relu')(path2)
        path2 = Dense(units=64, activation='relu')(path2)
        path2 = Dense(units=10, activation='softmax')(path2)

        attention_weights = Add()([path1, path2])
        attention_weights = Lambda(lambda x: K.softmax(x, axis=-1))(attention_weights)
        attention_weights = Multiply()([attention_weights, input_tensor])

        return attention_weights
    
    block1_output = block1(initial_conv)

    def block2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2))(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        cat = Concatenate()([avg_pool, max_pool])
        cat = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(cat)
        cat ='sigmoid')(cat)
        cat = Multiply()([cat, input_tensor])
        return cat

    block2_output = block2(block1_output)
    add = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(block2_output)
    add = Add()([add, block1_output])
    activation ='relu')(add)

    flatten_layer = Flatten()(activation)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model