import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input channels into three groups using Lambda
    def channel_split(input_tensor):
        # Split the input into three separate groups along the channel dimension
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(channel_split)(input_layer)
    
    # Applying different convolutional kernels to each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    
    # Concatenating the outputs from the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])
    
    # Flattening and passing through fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model