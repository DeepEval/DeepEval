import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    block_1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block_1_drop = Dropout(rate=0.25)(block_1_conv)
    block_1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_1_drop)
    block_1_output = keras.layers.add([block_1_conv2, input_layer])

    block_2_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_1_output)
    block_2_path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_2_split[0])
    block_2_path_1_drop = Dropout(rate=0.25)(block_2_path_1)
    block_2_path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_split[1])
    block_2_path_2_drop = Dropout(rate=0.25)(block_2_path_2)
    block_2_path_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block_2_split[2])
    block_2_path_3_drop = Dropout(rate=0.25)(block_2_path_3)
    block_2_concat = keras.layers.concatenate([block_2_path_1_drop, block_2_path_2_drop, block_2_path_3_drop])
    block_2_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_2_concat)
    block_2_output_drop = Dropout(rate=0.25)(block_2_output)
    block_2_output_flat = Flatten()(block_2_output_drop)

    dense = Dense(units=10, activation='softmax')(block_2_output_flat)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model