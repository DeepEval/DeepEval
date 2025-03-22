import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        dropout = Dropout(rate=0.2)(conv3)
        output_tensor = Concatenate()([conv1, conv2, dropout])
        return output_tensor

    def block_2(input_tensor):
        conv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        conv4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_tensor)
        conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')(input_tensor)
        conv6 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool3)
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model