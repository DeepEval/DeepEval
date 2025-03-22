import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, GlobalMaxPooling2D, Reshape, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    transition_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(block_1(input_tensor=input_layer))

    def block_2(input_tensor):
        global_max_pooling = GlobalMaxPooling2D()(input_tensor)
        dense1 = Dense(units=32)(global_max_pooling)
        dense2 = Dense(units=32)(dense1)
        weights = Reshape(target_shape=(32,))(dense2)
        output_tensor = Multiply()([weights, input_tensor])
        return output_tensor

    main_path = block_2(input_tensor=transition_conv)
    branch_path = input_layer
    adding_layer = Add()([main_path, branch_path])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model