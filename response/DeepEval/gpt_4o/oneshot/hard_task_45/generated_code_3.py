import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Splitting and depthwise separable convolutions
    def first_block(input_tensor):
        split_1, split_2, split_3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_2)
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_3)
        
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = first_block(input_layer)

    # Second block: Multiple branches for feature extraction
    def second_block(input_tensor):
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block2_output = second_block(block1_output)

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model