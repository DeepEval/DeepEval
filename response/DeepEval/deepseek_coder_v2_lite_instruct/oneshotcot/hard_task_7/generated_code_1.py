import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense, Permute, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Split the input into two groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(conv1)
    
    # Process the first group
    group1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1_depthwise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(split_layer[0])
    group1_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
    
    # Process the second group without modification
    group2 = split_layer[1]
    
    # Merge the outputs of both groups
    merged = Concatenate()([group1_conv1, group1_conv2, group2])
    
    # Block 2
    block2_input_shape = tf.keras.backend.int_shape(merged)
    reshaped = Reshape((block2_input_shape[1], block2_input_shape[2], 2, int(block2_input_shape[3]/2)))(merged)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    final_shape = tf.keras.backend.int_shape(permuted)
    reshaped_back = Reshape((final_shape[1], final_shape[2], final_shape[3] * final_shape[4]))(permuted)
    
    # Flatten the final output
    flattened = Flatten()(reshaped_back)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])