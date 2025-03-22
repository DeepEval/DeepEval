import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Lambda, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group1 = BatchNormalization()(group1)
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group2 = BatchNormalization()(group2)
        group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group3 = BatchNormalization()(group3)
        outputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(Add()([group1, group2, group3]))
        output_tensor = Concatenate()([outputs_groups[0], outputs_groups[1], outputs_groups[2]])
        return output_tensor

    def block_2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3, input_tensor])
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4, input_tensor])
        outputs_paths = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(Add()([path1, path2, path3, path4]))
        output_tensor = Concatenate()([outputs_paths[0], outputs_paths[1], outputs_paths[2], outputs_paths[3]])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model