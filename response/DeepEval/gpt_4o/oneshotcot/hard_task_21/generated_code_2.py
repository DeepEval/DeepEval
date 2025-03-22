import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from the three paths
    main_path_output = Concatenate()([path1, path2, path3])
    
    # Branch path: 1x1 convolution to match the output channel size of the main path
    branch_path = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the main and branch paths
    added_output = Add()([main_path_output, branch_path])
    
    # Flatten the result and add fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model