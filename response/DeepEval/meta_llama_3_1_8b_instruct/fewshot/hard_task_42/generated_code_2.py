import keras
from keras.layers import Input, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, Dropout, DepthwiseConv2D

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(0.2)(flatten1)

        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(0.2)(flatten2)

        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(0.2)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)

        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv1 = Flatten()(conv1)

        conv2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = Conv2D(kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv2 = Conv2D(kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        conv2 = Flatten()(conv2)

        conv3 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = Conv2D(kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv3 = Conv2D(kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv3 = Conv2D(kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        conv3 = Flatten()(conv3)

        conv4 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        conv4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        conv4 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
        conv4 = Flatten()(conv4)

        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(1, 1, 64))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model