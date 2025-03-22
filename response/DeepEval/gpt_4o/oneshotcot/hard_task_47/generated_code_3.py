import keras
from keras.layers import Input, Conv2D, SeparableConv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    
    def split_input(input_tensor):
        # Split the input tensor into three parts along the last dimension
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    split_layers = Lambda(split_input)(input_layer)
    
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    path1 = BatchNormalization()(path1)
    
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    path2 = BatchNormalization()(path2)
    
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
    path3 = BatchNormalization()(path3)
    
    block1_output = Concatenate()([path1, path2, path3])
    
    # Second Block
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: Average Pooling
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(block1_output)
    
    block2_output = Concatenate()([branch1, branch2, branch3])
    
    # Fully Connected Layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model