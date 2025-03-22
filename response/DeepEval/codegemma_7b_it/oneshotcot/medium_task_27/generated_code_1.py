import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch_conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    branch_concat = Concatenate()([branch_conv_3x3, branch_conv_5x5])

    global_avg_pool = GlobalAveragePooling2D()(branch_concat)

    branch_dense_1 = Dense(units=128, activation='relu')(global_avg_pool)
    branch_dense_2 = Dense(units=10, activation='softmax')(branch_dense_1)

    output_layer = tf.keras.layers.add([branch_dense_1, branch_dense_2])

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model