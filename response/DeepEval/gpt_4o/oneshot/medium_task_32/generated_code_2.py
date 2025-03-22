import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define the input layer with shape corresponding to CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Use a Lambda layer to split the input along the last dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define depthwise separable convolution layers for each split
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_channels[2])
    
    # Concatenate the outputs from the three convolutional operations
    concat_layer = Concatenate()([conv1, conv2, conv3])
    
    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concat_layer)
    
    # Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model with input and output layers
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model