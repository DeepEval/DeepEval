import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Concatenate, Lambda, AveragePooling2D, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = BatchNormalization()(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = BatchNormalization()(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(max_pooling)
        conv4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(max_pooling)
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv5])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model