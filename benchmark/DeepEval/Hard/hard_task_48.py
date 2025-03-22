import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Dense, Flatten

def dl_model():
    
    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):

        inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        batch_norm2 = BatchNormalization()(conv2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        batch_norm3 = BatchNormalization()(conv3)
        
        output_tensor = Concatenate()([batch_norm1, batch_norm2, batch_norm3])

        return output_tensor

    def block_2(input_tensor):

        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_tensor)

        pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

        conv_3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_3_2_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu')(conv_3_1)
        conv_3_2_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(2, 2), padding='same', activation='relu')(conv_3_1)
        path3 = Concatenate(axis=-1)([conv_3_2_1, conv_3_2_2])

        conv_4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_4_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_4_1)
        conv_4_3_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu')(conv_4_2)
        conv_4_3_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(2, 2), padding='same', activation='relu')(conv_4_2)
        path4 = Concatenate(axis=-1)([conv_4_3_1, conv_4_3_2])

        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)
    
    flatten = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
