import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_input = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(branch_conv)
    concat_input = Concatenate()([conv1, conv2, conv3, branch_pool])
    batch_norm = BatchNormalization()(concat_input)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model