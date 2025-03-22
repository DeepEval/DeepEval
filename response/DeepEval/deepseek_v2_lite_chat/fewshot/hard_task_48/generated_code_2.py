import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, BatchNormalization, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split input into three groups and process each group separately
    def block_1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)
        concat = Concatenate(axis=-1)([bn1, bn2, bn3])
        return concat

    # Block 2: Four parallel branches
    def block_2(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
        branch2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch3 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch4 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
        concat = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        flatten = Flatten()(concat)
        dense = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(dense)
        return output

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model