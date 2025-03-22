import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Input is split into three groups, each processed by a 1x1 convolutional layer
    def block1(input_tensor):
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        processed_splits = [Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in splits]
        output_tensor = Concatenate(axis=-1)(processed_splits)
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2: Reshape and permutation for channel shuffling
    reshaped = Reshape((-1, 3, int(block1_output.shape[3] / 3)))(block1_output)
    permuted = Permute((3, 1, 2))(reshaped)
    permuted = Reshape((-1, int(permuted.shape[1]), int(permuted.shape[2] * 3)))(permuted)
    
    # Block 3: 3x3 depthwise separable convolution
    depthwise_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', depthwise_mode=True, activation='relu')(permuted)
    
    # Branch from input
    branch = input_layer
    
    # Add the main path and the branch
    added = Add()([depthwise_conv, branch])
    
    # Flatten the final output
    flatten = Flatten()(added)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()