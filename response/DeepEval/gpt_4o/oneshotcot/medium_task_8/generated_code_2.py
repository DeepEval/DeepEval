import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    # Splitting the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = split_layer[0]
    
    # Second group with 3x3 convolution
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    
    # Combine second group with third group and add another 3x3 convolution
    combined_group = Concatenate()([group2, split_layer[2]])
    group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
    
    # Concatenate all three groups
    main_path_output = Concatenate()([group1, group2, group3])
    
    # Branch Path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Main Path and Branch Path outputs
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output and pass through a fully connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model