import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layers = Lambda(split_input)(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes
    group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([group1, group2, group3])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model