import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import  tf

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

        def block1(input_tensor):
            split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=3))(input_tensor)
            
            branch1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
            branch1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
            branch1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
            
            merged_tensor = Concatenate(axis=3)([branch1_1, branch1_2, split_tensor[1], branch1_3])
            
            return merged_tensor

        block1_output = block1(conv1)

        def block2(input_tensor):
            shape = Lambda(lambda x: tf.shape(x))(input_tensor)
            reshaped_tensor = Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], 4, input_tensor.shape[-1]//4]))(input_tensor)
            permuted_tensor = Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2, 4]))(reshaped_tensor)
            return Lambda(lambda x: tf.reshape(x, [-1, shape[1], shape[2], input_tensor.shape[-1]]) )(permuted_tensor)

        block2_output = block2(block1_output) 

        flatten_layer = Flatten()(block2_output)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model