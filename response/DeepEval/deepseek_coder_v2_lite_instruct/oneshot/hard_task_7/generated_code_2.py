import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer in Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Block 1
    split_layer = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv1)
    
    # First group operations
    group1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1_depthwise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(group1_conv1x1)
    group1_conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
    
    # Second group as is
    group2 = split_layer[1]
    
    # Concatenate outputs from both groups
    block1_output = Concatenate()([group1_conv1x1_2, group2])
    
    # Batch Normalization and Flattening
    batch_norm = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Block 2
    block2_input = Reshape((-1, 28, 28, 1))(flatten_layer)
    block2_input = Permute((1, 2, 4, 3))(block2_input)
    block2_input = Reshape((-1, 28 * 4, 28))(block2_input)
    
    # Flatten and pass through fully connected layers
    flatten_block2 = Flatten()(block2_input)
    dense1 = Dense(units=128, activation='relu')(flatten_block2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()