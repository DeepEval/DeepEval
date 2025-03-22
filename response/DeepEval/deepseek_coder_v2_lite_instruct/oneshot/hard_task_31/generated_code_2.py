import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        
        # Branch path
        branch_path = input_tensor
        
        # Add the outputs of both paths
        output_tensor = tf.add(main_path, branch_path)
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Second block
    def block2(input_tensor):
        # Split the input into three groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with separable convolutional layers
        paths = []
        for group in split_layer:
            path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
            path1 = Dropout(0.25)(path1)
            
            path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group)
            path2 = Dropout(0.25)(path2)
            
            path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group)
            path3 = Dropout(0.25)(path3)
            
            # Concatenate the outputs of the three paths
            path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tf.concat([path1, path2, path3], axis=-1))
            paths.append(path_output)
        
        # Concatenate the outputs of the three paths
        final_output = Concatenate()(paths)
        
        return final_output
    
    block2_output = block2(block1_output)
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model