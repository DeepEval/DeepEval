import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, DepthwiseConv2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)
        shortcut = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        addition = keras.layers.add([shortcut, conv3])
        return addition

    def block2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model