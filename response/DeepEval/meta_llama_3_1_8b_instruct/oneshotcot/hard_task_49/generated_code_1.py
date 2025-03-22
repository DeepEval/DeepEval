import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, DepthwiseConv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool1_flatten = Flatten()(pool1)
    
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool2_flatten = Flatten()(pool2)
    
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pool3_flatten = Flatten()(pool3)
    
    block_output = Concatenate()([pool1_flatten, pool2_flatten, pool3_flatten])
    
    # Transformation layer
    dense1 = Dense(units=128, activation='relu')(block_output)
    reshape_layer = Reshape((4, 128))(dense1)
    
    # Second block
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
    
    split_output = split_input(reshape_layer)
    
    path1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_output[2])
    path4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_output[3])
    
    output_tensor = Concatenate()([path1, path2, path3, path4])
    
    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model