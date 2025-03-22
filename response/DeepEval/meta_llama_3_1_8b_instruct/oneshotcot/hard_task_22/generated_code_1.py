import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import Add
from tensorflow.keras import regularizers
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: multi-scale feature extraction with separable convolutional layers
    conv_path1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path2 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path3 = Conv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge the three groups of feature maps along the channel dimension
    merge_path = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply separable convolutional layers to each group of feature maps
    conv_group1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merge_path[0])
    conv_group2 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge_path[1])
    conv_group3 = Conv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(merge_path[2])
    
    # Concatenate the outputs from the three groups of feature maps
    output_path = Concatenate()([conv_group1, conv_group2, conv_group3])
    
    # Branch path: align the number of output channels with those of the main path
    branch_path = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fusion_layer = Add()([output_path, branch_path])
    
    # Flatten the combined output into a one-dimensional vector
    flatten_layer = Flatten()(fusion_layer)
    
    # Apply two fully connected layers for a 10-class classification task
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model