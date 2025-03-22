import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Add, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        input_shape = keras.backend.int_shape(input_tensor)
        group_size = 3
        input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=group_size, axis=-1))(input_tensor)
        conv_outputs = []
        for i in range(group_size):
            conv_outputs.append(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[i]))
        return Concatenate()([conv_outputs[0], conv_outputs[1], conv_outputs[2]])

    main_path = block(input_tensor=input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    fused_output = Add()([main_path, branch_path])
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model