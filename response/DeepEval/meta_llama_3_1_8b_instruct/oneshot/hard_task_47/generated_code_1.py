import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, AveragePooling2D, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the last dimension
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    
    split_output = split_input(input_layer)
    
    # Apply depthwise separable convolutional layers with different kernel sizes for each group
    conv1x1 = Lambda(lambda x: [DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(x[i]) for i in range(3)])(split_output)
    conv3x3 = Lambda(lambda x: [DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x[i]) for i in range(3)])(split_output)
    conv5x5 = Lambda(lambda x: [DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(x[i]) for i in range(3)])(split_output)
    
    # Concatenate the outputs from these three groups
    concat_output = Concatenate()([conv1x1[0], conv3x3[0], conv5x5[0],
                                   conv1x1[1], conv3x3[1], conv5x5[1],
                                   conv1x1[2], conv3x3[2], conv5x5[2]])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(concat_output)
    
    # Define the branches for feature extraction
    branch1 = Conv2D(kernel_size=(1, 1), activation='relu')(batch_norm)
    branch2 = Conv2D(kernel_size=(1, 1), activation='relu')(batch_norm)
    branch2 = Conv2D(kernel_size=(1, 7), activation='relu')(branch2)
    branch2 = Conv2D(kernel_size=(7, 1), activation='relu')(branch2)
    branch2 = Conv2D(kernel_size=(3, 3), activation='relu')(branch2)
    branch3 = AveragePooling2D(pool_size=(2, 2))(batch_norm)
    
    # Concatenate the outputs from all branches
    concat_output = Concatenate()([branch1, branch2, branch3])
    
    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Apply two fully connected layers to produce the final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model