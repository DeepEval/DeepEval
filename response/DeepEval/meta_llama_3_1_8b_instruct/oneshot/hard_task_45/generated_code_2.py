import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import MaxPooling2D, Add, Multiply
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)
    
    split_input_layer = Lambda(split_input)(input_layer)
    
    # Each group employs depthwise separable convolutional layers with varying kernel sizes
    group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input_layer[2])
    
    # Concatenate the outputs from these groups
    output_tensor = Concatenate()([group1, group2, group3])
    
    # Second Block
    branch1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch2 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch2 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch3 = Conv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output_tensor)
    branch4 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    
    # Finally, the output should pass through a fully connected layer to produce the final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model