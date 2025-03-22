import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Multiply
from keras.layers import GlobalMaxPooling2D, Add
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(split_layer[0])
    block1_output = Concatenate()([block1_output, block1(split_layer[1]), block1(split_layer[2])])
    
    transition_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)
    
    # Block 2
    max_pooling = GlobalMaxPooling2D()(transition_conv)
    weights1 = Dense(96, activation='relu')(max_pooling)
    weights2 = Dense(48, activation='relu')(weights1)
    reshape_layer = Reshape((1, 1, 48))(weights2)
    output = Multiply()([reshape_layer, transition_conv])
    
    branch = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output = Add()([output, branch])
    
    output = Dense(10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)

    return model