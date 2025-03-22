import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: split input into three groups and apply depthwise separable convolutions
    def depthwise_separable_block(input_tensor):
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Each split applies a different kernel size
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', depth_multiplier=1)(split_tensors[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depth_multiplier=1)(split_tensors[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depth_multiplier=1)(split_tensors[2])
        
        # Concatenate outputs
        return Concatenate()([path1, path2, path3])
    
    block1_output = depthwise_separable_block(input_layer)

    # Second block: multiple branches for feature extraction
    def multi_branch_block(input_tensor):
        # Each branch processes the input separately
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        
        path4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        
        # Concatenate outputs
        return Concatenate()([path1, path2, path3, path4])

    block2_output = multi_branch_block(block1_output)

    # Final layers: flattening and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model