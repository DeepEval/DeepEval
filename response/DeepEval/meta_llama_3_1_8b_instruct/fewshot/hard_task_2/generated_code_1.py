import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def group_operation(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = conv3
        return output_tensor

    group1_output = group_operation(input_tensor=input_layer)
    group2_output = group_operation(input_tensor=input_layer)
    group3_output = group_operation(input_tensor=input_layer)

    adding_layer1 = Add()([group1_output, group2_output, group3_output])
    adding_layer2 = Add()([adding_layer1, input_layer])

    flatten_layer = Flatten()(adding_layer2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model