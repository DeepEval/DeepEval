import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        concat_main = Concatenate()([conv1, conv2, conv3])
        return concat_main
    
    def branch_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    
    concat_output = Add()([main_path_output, branch_path_output])
    flatten_output = Flatten()(concat_output)
    dense1 = Dense(units=64, activation='relu')(flatten_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model