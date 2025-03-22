import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Create the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Use a Lambda layer to split the input along the last dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 3: Define a function to apply depthwise separable convolutions
    def depthwise_sep_conv(input_tensor, kernel_size):
        return Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), 
                      padding='same', activation='relu', 
                      depthwise=True)(input_tensor)

    # Step 4: Apply the depthwise separable convolutions to each split
    path1 = depthwise_sep_conv(split_tensors[0], kernel_size=1)
    path2 = depthwise_sep_conv(split_tensors[1], kernel_size=3)
    path3 = depthwise_sep_conv(split_tensors[2], kernel_size=5)

    # Step 5: Concatenate the outputs from the three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Step 6: Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Step 7: Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 8: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model