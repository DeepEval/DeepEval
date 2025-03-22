import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense, Reshape, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Path 1: max pooling with 1x1
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(0.2)(flatten1)

        # Path 2: max pooling with 2x2
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(0.2)(flatten2)

        # Path 3: max pooling with 4x4
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(0.2)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 + 1x7 + 7x1 convolution
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv4)
        conv6 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv5)
        concat2 = Concatenate()([conv2, conv5, conv6])

        # Path 3: 1x1 + 7x1 + 1x7 + 7x1 convolution
        conv7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv8 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv7)
        conv9 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv8)
        conv10 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv9)
        conv11 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv10)
        conv12 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv11)
        concat3 = Concatenate()([conv7, conv8, conv9, conv10, conv11, conv12])

        # Path 4: 1x1 convolution + average pooling
        conv13 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv13)
        concat4 = Concatenate()([conv13, maxpool4])

        output_tensor = Concatenate()([conv1, concat2, concat3, concat4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 64))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model