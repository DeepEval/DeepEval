import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
        concat = Concatenate()( [conv1, conv2, conv3] )
        return concat

    def block_2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch7 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch8 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        concat = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8])
        return concat

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=1000, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model