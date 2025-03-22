import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
        bath_norm = BatchNormalization()(Concatenate()([conv1, conv2, conv3]))
        return bath_norm
    
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor))
        conv_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        split_path3 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1))
        conv_path3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(split_path3[0])
        conv_path4 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(split_path3[1])
        conv_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_path4)
        split_path4 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1))
        conv_path5 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(split_path4[0])
        conv_path6 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(split_path4[1])
        conv_path6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_path6)
        concat_layer = Concatenate()([path1, conv_path2, Concatenate()([conv_path3, conv_path4]), Concatenate()([conv_path5, conv_path6])])
        return concat_layer
    
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model