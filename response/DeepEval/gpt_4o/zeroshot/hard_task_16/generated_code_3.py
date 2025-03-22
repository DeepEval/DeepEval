import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, GlobalMaxPooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images shape

    # Block 1: Splitting input into three groups
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_input)(input_layer)
    
    # Each group gets a series of convolutions: 1x1 -> 3x3 -> 1x1
    def conv_block(x):
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        return x
    
    conv_group1 = conv_block(split_layer[0])
    conv_group2 = conv_block(split_layer[1])
    conv_group3 = conv_block(split_layer[2])
    
    # Concatenate all groups
    concatenated = Concatenate(axis=-1)([conv_group1, conv_group2, conv_group3])
    
    # Transition Convolution: Adjusting the number of channels to match the input layer
    transition_conv = Conv2D(3, (1, 1), activation='relu')(concatenated)

    # Block 2: Global Max Pooling
    pooled = GlobalMaxPooling2D()(transition_conv)
    
    # Fully Connected layers to generate channel-matching weights
    fc1 = Dense(64, activation='relu')(pooled)
    fc2 = Dense(3, activation='sigmoid')(fc1)  # 3 channels to match input
    
    # Reshape weights to match adjusted output shape
    reshaped_weights = tf.reshape(fc2, (-1, 1, 1, 3))
    
    # Multiply weights with adjusted output (element-wise)
    weighted_output = Multiply()([transition_conv, reshaped_weights])
    
    # Branch directly connected to input
    branch_output = input_layer
    
    # Add main path and branch output
    added_output = Add()([weighted_output, branch_output])
    
    # Fully connected layer for classification
    flatten = Flatten()(added_output)
    output_layer = Dense(10, activation='softmax')(flatten)  # 10 classes in CIFAR-10
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model