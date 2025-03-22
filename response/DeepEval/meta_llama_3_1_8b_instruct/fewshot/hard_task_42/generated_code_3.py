import keras
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, Dropout, Dense, Reshape, Conv2D

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(0.2)(flatten1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(0.2)(flatten2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(0.2)(flatten3)
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)

        # Path 1: Single 1x1 Convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        maxpool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        flatten1 = Flatten()(maxpool1)

        # Path 2: 1x1 Convolution followed by 1x7 Convolution and 7x1 Convolution
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        flatten2 = Flatten()(maxpool2)

        # Path 3: 1x1 Convolution followed by Alternating 7x1 and 1x7 Convolutions
        conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv6 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv5)
        conv7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv6)
        conv8 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv7)
        conv9 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv8)
        maxpool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv9)
        flatten3 = Flatten()(maxpool3)

        # Path 4: Average Pooling with 1x1 Convolution
        conv10 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        maxpool4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv10)
        flatten4 = Flatten()(maxpool4)

        output_tensor = Concatenate()([flatten1, flatten2, flatten3, flatten4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 64))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=32, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model