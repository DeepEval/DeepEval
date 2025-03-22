import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense, MaxPooling2D, Permute

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
        
        def block1(input_tensor):
            
            split_tensor = Lambda(lambda x: tf.split(x, 2, axis=3))(input_tensor)
            
            path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
            path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
            path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)
            
            path2 = split_tensor[1]
            
            merged_tensor = Concatenate()([path1, path2])
            
            return merged_tensor
        
        block1_output = block1(conv)
        
        def block2(input_tensor):
            shape = Lambda(lambda x: tf.shape(x))(input_tensor)
            
            reshaped_tensor = Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], input_tensor.shape[-1] // 4]))(input_tensor)
            
            permuted_tensor = Permute((1, 2, 3, 4))(reshaped_tensor)
            
            reshaped_back = Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], input_tensor.shape[-1]]) )(permuted_tensor)
            return reshaped_back
        
        block2_output = block2(block1_output)

        flatten_layer = Flatten()(block2_output)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model