import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block1(input_tensor):
        splitted_input = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        conv_outputs = []
        for x in splitted_input:
            conv_outputs.append(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))
        concat_output = Concatenate(axis=-1)(conv_outputs)
        return concat_output

    block1_output = block1(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model