import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda
from keras.layers import DepthwiseConv2D, Reshape
import tensorflow as tf

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: First block with average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flatten the pooled outputs
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    block1_output = Concatenate()([flat1, flat2, flat3])

    # Step 3: Fully connected layer before the second block
    dense1 = Dense(units=128, activation='relu')(block1_output)
    
    # Step 4: Reshape for second block (into a 4D tensor)
    reshaped = Reshape((1, 1, 128))(dense1)

    # Step 5: Second block with depthwise separable convolutions
    def split_and_process(input_tensor):
        split_tensors = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        depthwise_outputs = []
        for i, tensor in enumerate(split_tensors):
            kernel_size = (1 + 2 * i, 1 + 2 * i)  # Create kernel sizes 1x1, 3x3, 5x5, 7x7
            depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='relu')(tensor)
            depthwise_outputs.append(depthwise_conv)
        return Concatenate()(depthwise_outputs)

    block2_output = Lambda(split_and_process)(reshaped)

    # Step 6: Flatten the output of the second block
    flat2 = Flatten()(block2_output)

    # Step 7: Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(flat2)

    # Step 8: Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model