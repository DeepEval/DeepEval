import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    block1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def conv_layer(input_tensor):
        conv = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    conv1 = conv_layer(block1[0])
    conv2 = conv_layer(block1[1])
    conv3 = conv_layer(block1[2])
    fused_features = Concatenate()([conv1, conv2, conv3])
    
    block2 = Lambda(lambda x: tf.shape(x))(fused_features)
    reshaped = Lambda(lambda x: tf.reshape(x, shape=(-1, x.shape[1], x.shape[2], 3, int(x.shape[-1] / 3))))(fused_features)
    permuted = Lambda(lambda x: tf.transpose(x, (0, 1, 2, 4, 3)))(reshaped)
    reshaped_back = Lambda(lambda x: tf.reshape(x, shape=(-1, x.shape[1], x.shape[2], int(x.shape[-1] / 3) * 3)))(permuted)
    
    block3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_back)
    
    branch_path = Conv2D(filters=int(input_layer.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    adding_layer = Add()([block3, branch_path])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model