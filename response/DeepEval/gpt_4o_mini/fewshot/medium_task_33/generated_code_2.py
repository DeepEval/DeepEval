import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split the input into three channel groups
    def split_channels(input_tensor):
        return Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Define the feature extraction block
    def feature_extraction(input_tensor):
        channel_groups = split_channels(input_tensor)
        
        # Apply depthwise separable convolutions to each group
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(channel_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(channel_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(channel_groups[2])
        
        # Concatenate the outputs from the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Apply feature extraction to the input layer
    feature_output = feature_extraction(input_layer)
    
    # Flatten the concatenated outputs and pass through fully connected layers
    flatten_layer = Flatten()(feature_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model