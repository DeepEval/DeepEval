import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense, tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        return Concatenate()([conv1_1, conv2_1, conv3_1])

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    combined_output = Add()([main_path(input_layer), branch_path])
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model