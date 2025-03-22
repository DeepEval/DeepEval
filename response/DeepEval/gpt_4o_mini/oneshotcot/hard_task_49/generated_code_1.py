import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Concatenate, DepthwiseConv2D, Reshape
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Average Pooling Layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs from pooling layers
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully Connected Layer
    dense_layer = Dense(units=128, activation='relu')(concatenated)

    # Reshape into 4D Tensor
    reshaped = Reshape((1, 1, 128))(dense_layer)

    # Second Block: Depthwise Separable Convolutions
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Applying Depthwise Separable Convolution on each split
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
    path4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_tensor[3])

    # Concatenate the outputs of the depthwise separable convolutions
    concatenated_depthwise = Concatenate()([path1, path2, path3, path4])

    # Flatten the final output
    flatten_output = Flatten()(concatenated_depthwise)

    # Fully Connected Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model