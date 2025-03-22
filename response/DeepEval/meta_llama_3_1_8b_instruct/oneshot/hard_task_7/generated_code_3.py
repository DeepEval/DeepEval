import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    def block1(input_tensor):
        split_layer = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        dw_conv = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        output_tensor = layers.Concatenate()([conv1, conv2, split_layer[1]])

        return output_tensor
    
    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped_tensor = layers.Reshape((shape[1], shape[2], shape[3]//2, 2))(input_tensor)
        permuted_tensor = layers.Permute((1, 2, 4, 3))(reshaped_tensor)
        reshaped_tensor = layers.Reshape((shape[1], shape[2], shape[3]))(permuted_tensor)
        
        return reshaped_tensor
    
    block1_output = block1(conv)
    block2_output = block2(block1_output)
    flatten_layer = layers.Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model