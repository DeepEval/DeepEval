import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path: Splitting input into three groups
    def main_path(input_tensor):
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    # Branch Path: Processing with 1x1 Convolution
    def branch_path(input_tensor):
        branch_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_conv

    # Paths Output
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    
    # Combining the outputs from main path and branch path
    fused_features = Add()([main_path_output, branch_path_output])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model