import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Lambda, DepthwiseConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    reshape_layer = Reshape((3, 1, 1))(concatenated)  # Transform into 4-dimensional tensor
    
    # Second Block
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)
    
    outputs = []
    for i, split in enumerate(split_layer):
        if i == 0:
            conv = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split)
        elif i == 1:
            conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split)
        elif i == 2:
            conv = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split)
        elif i == 3:
            conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split)
        
        outputs.append(conv)
    
    concatenated_outputs = Concatenate(axis=-1)(outputs)
    flatten_layer = Flatten()(concatenated_outputs)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model