import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Split and Depthwise Separable Convolutions
    def split_block(input_tensor):
        # Split the input into three groups along the last dimension
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = split_block(input_layer)

    # Second Block: Multiple Branches with Feature Extraction
    def feature_extraction_block(input_tensor):
        # Different configurations for each branch
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block2_output = feature_extraction_block(block1_output)

    # Flatten and Fully Connected Layer for Output
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model