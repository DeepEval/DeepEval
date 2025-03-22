import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block with three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each average pooling result
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    # Concatenate flattened results
    concat1 = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer after first block and reshape
    fc1 = Dense(units=784, activation='relu')(concat1)
    reshape1 = Reshape((28, 28, 1))(fc1)

    # Second block with tf.split and depthwise separable convolutions
    # Split input into four groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape1)

    # Apply depthwise separable convolutions with different kernel sizes
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_layer[3])

    # Concatenate the outputs of the depthwise separable convolutions
    concat2 = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten and apply a fully connected layer for classification
    flat_final = Flatten()(concat2)
    output_layer = Dense(units=10, activation='softmax')(flat_final)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model