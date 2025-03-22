import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Add, DepthwiseConv2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32,32,3)) 

    def block_1(input_tensor):
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        main_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)

        output_tensor = Add()([main_path, input_tensor])

        return output_tensor

    def block_2(input_tensor):

        inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        concatenate = Concatenate()([conv1, conv2, conv3])
        output_tensor = concatenate

        return output_tensor

    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)

    dense1= Dense(units=128, activation='relu')(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
