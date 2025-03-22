import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, Lambda, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout = Dropout(0.2)(conv)
        conv_restored = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)
        branch_path = input_tensor
        adding_layer = Add()([conv_restored, branch_path])
        return adding_layer

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout1 = Dropout(0.2)(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        dropout2 = Dropout(0.2)(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        dropout3 = Dropout(0.2)(conv3)
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model