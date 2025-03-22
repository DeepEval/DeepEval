import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        return maxpool
    
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    group1_output = block(group1)
    
    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    group2_output = block(group2)
    
    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    group3_output = block(group3)
    
    concat = Concatenate(axis=3)([group1_output, group2_output, group3_output])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model