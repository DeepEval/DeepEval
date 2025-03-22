import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # 1x1 convolution
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 2x2 convolution
        conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 4x4 convolution
        conv3 = Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # average pooling
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(conv3)
        # flatten
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        # concatenate
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        # 1x1 convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 convolution
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 7x7 convolution
        conv3 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x7 convolution
        conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 convolution
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # average pooling
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv2)
        pool3 = MaxPooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(conv3)
        pool4 = MaxPooling2D(pool_size=(1, 7), strides=(1, 1), padding='same')(conv4)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv5)
        # flatten
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        flatten4 = Flatten()(pool4)
        flatten5 = Flatten()(pool5)
        # concatenate
        output_tensor = Concatenate()([flatten1, flatten2, flatten3, flatten4, flatten5])
        return output_tensor

    # 1st block
    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    # 2nd block
    block2_output = block_2(input_tensor=reshaped)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model