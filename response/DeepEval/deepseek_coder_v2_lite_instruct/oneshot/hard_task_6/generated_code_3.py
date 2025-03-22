import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split into 3 groups, apply 1x1 Conv, and concatenate
    def block1(x):
        splits = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=-1))(x)
        conv_groups = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        return Concatenate(axis=-1)(conv_groups)
    
    # Apply Block 1 to the input
    block1_output = block1(input_layer)
    
    # Block 2: Reshape, permute, and reshape back to apply channel shuffling
    def block2(x):
        shape = tf.keras.backend.int_shape(x)
        x = Reshape((shape[1], shape[2], 3, int(shape[3]/3)))(x)
        x = Permute((1, 2, 4, 3))(x)
        return Reshape((shape[1], shape[2], shape[3]))(x)
    
    # Apply Block 2 to the output of Block 1
    block2_output = block2(block1_output)
    
    # Block 3: Apply 3x3 depthwise separable convolution
    block3_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(block2_output)
    
    # Repeat Block 1
    block1_repeated_output = block1(input_layer)
    
    # Branch path: Apply average pooling
    branch_output = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    branch_output = Flatten()(branch_output)
    
    # Concatenate main path and branch path outputs
    concatenated_output = Concatenate(axis=-1)([block3_output, branch_output])
    
    # Fully connected layers
    fc1 = Dense(units=256, activation='relu')(concatenated_output)
    fc2 = Dense(units=128, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()