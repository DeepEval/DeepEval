import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Split the input into two groups
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv_initial)
    
    # First group operations
    group1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1_depthwise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(group1_1x1)
    group1_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
    
    # Second group remains unchanged
    group2 = split_layer[1]
    
    # Concatenate the outputs of both groups
    concatenated = Concatenate()([group1_1x1_2, group2])
    
    # Block 2
    block1_output = concatenated
    
    # Reshape and shuffle channels
    block1_output_shape = tf.keras.backend.int_shape(block1_output)
    reshaped = Reshape((block1_output_shape[1], block1_output_shape[2], 2, int(block1_output_shape[3]/2)))(block1_output)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    final_shape = tf.keras.backend.int_shape(permuted)
    reshaped_back = Reshape((final_shape[1], final_shape[2], final_shape[3] * final_shape[4]))(permuted)
    
    # Flatten and pass through fully connected layers
    flattened = Flatten()(reshaped_back)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()