import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Initial Convolution
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Splitting the input into two groups
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv1)
    
    # First group operations
    group1_conv1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1_dsconv = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(group1_conv1)
    group1_conv2 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_dsconv)
    
    # Second group operations
    group2_input = split_layer[1]
    
    # Merge the outputs from both groups
    merged = Concatenate()([group1_conv2, group2_input])
    
    # Block 2: Reshape, Permute, and Flatten
    reshape_layer = Reshape(target_shape=(int(merged.shape[1]), int(merged.shape[2]), 2, int(merged.shape[3])/2))(merged)
    permuted = Permute((1, 2, 4, 3))(reshape_layer)
    reshape_back = Reshape(target_shape=(int(permuted.shape[1]), int(permuted.shape[2]), int(permuted.shape[3]*2)))(permuted)
    
    # Flatten the final output
    flatten_layer = Flatten()(reshape_back)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()