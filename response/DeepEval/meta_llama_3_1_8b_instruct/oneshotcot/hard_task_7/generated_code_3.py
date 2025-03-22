import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def block1(input_tensor):
        split_axis = -1
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=split_axis))(input_tensor)
        
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', groups=32)(path1)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        path3 = input_groups[1]
        
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor

    block1_output = block1(conv)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)

    def block2(input_tensor):
        shape = K.int_shape(input_tensor)
        input_groups = Lambda(lambda x: tf.reshape(x, (-1, shape[1], shape[2], shape[3]//2, 2)))(input_tensor)
        
        permute = Permute((2, 3, 4, 1))(input_groups)
        input_groups = Lambda(lambda x: tf.reshape(x, (-1, shape[1], shape[2], shape[3], 2)))(permute)
        
        channel_shuffle_output = input_groups
        
        return channel_shuffle_output

    block2_output = block2(max_pooling)
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model