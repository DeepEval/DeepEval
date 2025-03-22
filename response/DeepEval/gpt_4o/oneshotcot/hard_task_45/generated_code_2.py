import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, MaxPooling2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split and apply depthwise separable convolutions
    def first_block(input_tensor):
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        sep_conv_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
        sep_conv_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
        sep_conv_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
        
        output_tensor = Concatenate()([sep_conv_1x1, sep_conv_3x3, sep_conv_5x5])
        return output_tensor

    first_block_output = first_block(input_layer)
    
    # Second block: Different branches for feature extraction
    def second_block(input_tensor):
        branch1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_1)
        branch2_3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_2)
        
        branch3_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3_1)
        
        branch4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4_2 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4_1)
        
        output_tensor = Concatenate()([branch1, branch2_3, branch3_2, branch4_2])
        return output_tensor
    
    second_block_output = second_block(first_block_output)
    
    # Final layers: Flatten and fully connected layer
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model