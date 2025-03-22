import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Split the input into three groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv_layers = []
    for group in split_inputs:
        # Number of filters is set to one-third of the input channels
        num_filters = 3  # Since the original input has 3 channels, we use 1 filter for each group
        conv_layer = Conv2D(filters=num_filters, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv_layers.append(conv_layer)

    # Apply average pooling to each convolutional layer
    pooled_layers = [AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer) for conv_layer in conv_layers]
    
    # Concatenate the pooled layers along the channel dimension
    concatenated_output = Concatenate(axis=-1)(pooled_layers)
    
    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model