import keras
from keras.layers import Input, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Concatenate, DepthwiseConv2D
from keras.models import Model


def dl_model():

    input_layer = Input(shape=(32, 32, 3))


    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor


    dropout_layer = Dropout(0.2)(output_tensor)


    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(dropout_layer)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor


    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)


    model = Model(inputs=input_layer, outputs=output_layer)


    return model