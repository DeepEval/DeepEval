import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Increase the number of channels using a 1x1 convolution
    conv1x1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)

    # Step 3: Compute channel attention weights
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=depthwise_conv.shape[-1], activation='sigmoid')(dense1)

    # Reshape the weights to match the initial features
    reshaped_weights = Reshape((1, 1, depthwise_conv.shape[-1]))(dense2)

    # Step 4: Apply the channel attention weights to the initial features
    attention_output = Multiply()([depthwise_conv, reshaped_weights])

    # Step 5: Reduce dimensionality with a 1x1 convolution
    reduced_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(attention_output)

    # Step 6: Combine the output with the initial input
    combined_output = keras.layers.Add()([input_layer, reduced_output])

    # Step 7: Flatten the output and apply a fully connected layer for classification
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model