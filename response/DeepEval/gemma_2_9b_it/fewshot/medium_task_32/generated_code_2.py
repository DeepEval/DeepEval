import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    
    merged_tensor = Concatenate()([conv1, conv2, conv3])
    
    flatten_layer = Flatten()(merged_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model