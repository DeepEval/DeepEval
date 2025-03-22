import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def split_channels(input_tensor):
        split_tensors = tf.split(input_tensor, num_or_size_splits=3, axis=2)
        return split_tensors
    
    split_output = Lambda(split_channels)(input_layer)
    
    group1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    group2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    group3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[2])
    
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group3)
    
    concat_output = Concatenate()([pool1, pool2, pool3])
    
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model