import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv, conv3, maxpool])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model