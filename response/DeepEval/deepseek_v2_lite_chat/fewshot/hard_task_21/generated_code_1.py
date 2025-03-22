import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Reshape, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat1

    def block_2(input_tensor):
        dense1 = Dense(units=64, activation='relu')(input_tensor)
        reshaped = Reshape(target_shape=(4, 4, 4))(dense1)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped)
        add_layer = Add()([conv1, conv2, conv3])
        flatten = Flatten()(add_layer)
        dense2 = Dense(units=128, activation='relu')(flatten)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        return output_layer

    main_path_output = block_1(input_tensor=input_layer)
    branch_path_output = block_2(input_tensor=main_path_output)
    model = keras.Model(inputs=input_layer, outputs=branch_path_output)

    return model

model = dl_model()
model.summary()