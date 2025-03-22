import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels with 3 channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input into three groups along the channel
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path: applying depthwise separable convolutions with different kernel sizes
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
    
    # Concatenating the outputs from the main path
    main_path_output = Concatenate()([path1, path2, path3])
    
    # Branch path: applying a 1x1 convolution to align channels
    branch_path_output = Conv2D(filters=tf.shape(main_path_output)[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding outputs from the main path and branch path
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flattening the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model