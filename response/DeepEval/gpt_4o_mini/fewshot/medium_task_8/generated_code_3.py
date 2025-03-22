import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        group1 = inputs_groups[0]  # First group (unchanged)
        
        # Second group with 3x3 convolution
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        
        # Combine second and third group
        combined_group = Add()([group2, inputs_groups[2]])
        
        # Another 3x3 convolution
        main_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
        
        # Concatenate all three groups
        output_tensor = Concatenate()([group1, main_output, inputs_groups[2]])
        
        return output_tensor
    
    # Branch path
    def branch_path(input_tensor):
        branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output
    
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    
    # Fuse both paths together using addition
    combined_output = Add()([main_output, branch_output])
    
    # Flatten the combined output and apply a fully connected layer
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model