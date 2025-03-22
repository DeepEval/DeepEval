import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Add, Lambda, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu', use_bias=False)(input_layer)
    bn = BatchNormalization()(conv)

    def block(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', use_bias=False)(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', use_bias=False)(conv2)
        output_tensor = Add()([conv1, conv3])
        return output_tensor

    block_output = block(input_tensor=bn)
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model