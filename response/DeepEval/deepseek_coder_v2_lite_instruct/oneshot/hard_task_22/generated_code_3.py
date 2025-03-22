import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(x):
        # 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        # 3x3 convolution
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depthwise_constraint=None)(x)
        # 5x5 convolution
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depthwise_constraint=None)(x)
        return path1, path2, path3
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    path1, path2, path3 = main_path(split_layer[0]), main_path(split_layer[1]), main_path(split_layer[2])
    
    # Concatenate the outputs from the three groups
    concatenated_output = Concatenate(axis=-1)([path1[0], path2[0], path3[0]])
    
    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fused_output = keras.layers.add([concatenated_output, branch_path])
    
    # Flatten the combined output
    flattened_layer = Flatten()(fused_output)
    
    # Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model