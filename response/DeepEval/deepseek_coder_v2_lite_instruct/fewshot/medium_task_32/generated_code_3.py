import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def split_and_process(input_tensor):
        # Split the input into three groups along the last dimension
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with a depthwise separable convolutional layer
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        
        # Concatenate the outputs of the three groups
        concatenated = Concatenate()([conv1, conv2, conv3])
        
        return concatenated

    # Apply the split_and_process function to the input
    processed_output = split_and_process(input_tensor=input_layer)

    # Flatten the concatenated output
    flattened = Flatten()(processed_output)

    # Pass the flattened output through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model