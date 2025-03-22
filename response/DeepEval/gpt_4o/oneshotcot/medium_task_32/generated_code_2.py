import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define the input layer with the shape of CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define depthwise separable convolutions with different kernel sizes
    group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs of the three groups
    concatenated_layer = Concatenate()([group1, group2, group3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_layer)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model